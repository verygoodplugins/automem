// Minimal MCP-over-SSE server that bridges to AutoMem HTTP API
// Exposes:
//   GET  /mcp/sse       -> SSE stream (clients POST JSON-RPC to /mcp/messages)
//   POST /mcp/messages  -> Accepts JSON-RPC messages for active sessions
//   GET  /health        -> Health probe

import express from 'express';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';

// Simple AutoMem HTTP client (mirrors the npm package behavior but inline to avoid version conflicts)
class AutoMemClient {
  constructor(config) {
    this.config = config;
  }
  async _request(method, path, body) {
    const url = `${this.config.endpoint.replace(/\/$/, '')}/${path.replace(/^\//, '')}`;
    const headers = { 'Content-Type': 'application/json' };
    if (this.config.apiKey) headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    console.log(`[AutoMem] ${method} ${url}`);
    let res;
    try {
      res = await fetch(url, {
        method,
        headers,
        body: method === 'GET' ? undefined : (body ? JSON.stringify(body) : undefined)
      });
    } catch (fetchErr) {
      console.error(`[AutoMem] Fetch failed for ${url}:`, fetchErr.message, fetchErr.cause);
      throw new Error(`fetch failed: ${fetchErr.message} (endpoint: ${this.config.endpoint})`);
    }
    let data;
    try {
      data = await res.json();
    } catch (_) {
      data = { message: await res.text() };
    }
    if (!res.ok) throw new Error(data.message || data.detail || `HTTP ${res.status}`);
    return data;
  }
  async storeMemory(args) {
    const body = {
      content: args.content,
      tags: args.tags || [],
      importance: args.importance,
      embedding: args.embedding,
      metadata: args.metadata,
      timestamp: args.timestamp,
      type: args.type,
      confidence: args.confidence,
      id: args.id,
      t_valid: args.t_valid,
      t_invalid: args.t_invalid,
      updated_at: args.updated_at,
      last_accessed: args.last_accessed
    };
    const r = await this._request('POST', 'memory', body);
    return { memory_id: r.memory_id || r.id, message: r.message || 'Memory stored successfully' };
  }
  async recallMemory(args) {
    const p = new URLSearchParams();
    if (args.query) p.set('query', args.query);
    if (args.limit) p.set('limit', String(args.limit));
    if (args.per_query_limit) p.set('per_query_limit', String(args.per_query_limit));
    if (Array.isArray(args.queries)) args.queries.forEach(q => { if (q) p.append('queries', q); });
    if (Array.isArray(args.embedding)) p.set('embedding', args.embedding.join(','));
    if (args.time_query) p.set('time_query', args.time_query);
    if (args.start) p.set('start', args.start);
    if (args.end) p.set('end', args.end);
    if (Array.isArray(args.tags)) args.tags.forEach(t => p.append('tags', t));
    if (args.tag_mode) p.set('tag_mode', args.tag_mode);
    if (args.tag_match) p.set('tag_match', args.tag_match);
    const path = p.toString() ? `recall?${p.toString()}` : 'recall';
    const r = await this._request('GET', path);
    return r;
  }
  async associateMemories(args) {
    const r = await this._request('POST', 'associate', {
      memory1_id: args.memory1_id,
      memory2_id: args.memory2_id,
      type: args.type,
      strength: args.strength
    });
    return { success: true, message: r.message || 'Association created successfully' };
  }
  async updateMemory(args) {
    const { memory_id, ...updates } = args;
    const r = await this._request('PATCH', `memory/${memory_id}`, updates);
    return { memory_id: r.memory_id || memory_id, message: r.message || 'Memory updated successfully' };
  }
  async deleteMemory(args) {
    const r = await this._request('DELETE', `memory/${args.memory_id}`);
    return { memory_id: r.memory_id || args.memory_id, message: r.message || 'Memory deleted successfully' };
  }
  async checkHealth() {
    try {
      const r = await this._request('GET', 'health');
      return r;
    } catch (e) {
      return { status: 'error', error: e.message };
    }
  }
}

// Build a new MCP Server instance with AutoMem tool handlers
function buildMcpServer(client) {
  const server = new Server({ name: 'automem-mcp-sse', version: '0.1.0' }, { capabilities: { tools: {} } });

  const tools = [
    {
      name: 'store_memory',
      description: 'Store a memory with optional tags, importance, metadata, timestamps, and embedding',
      inputSchema: {
        type: 'object',
        properties: {
          content: { type: 'string' },
          tags: { type: 'array', items: { type: 'string' } },
          importance: { type: 'number', minimum: 0, maximum: 1 },
          embedding: { type: 'array', items: { type: 'number' } },
          metadata: { type: 'object' },
          timestamp: { type: 'string' },
          type: { type: 'string' },
          confidence: { type: 'number', minimum: 0, maximum: 1 },
          id: { type: 'string' },
          t_valid: { type: 'string' },
          t_invalid: { type: 'string' },
          updated_at: { type: 'string' },
          last_accessed: { type: 'string' }
        },
        required: ['content']
      }
    },
    {
      name: 'recall_memory',
      description: 'Recall memories with hybrid semantic/keyword search and optional time/tag filters',
      inputSchema: {
        type: 'object',
        properties: {
          query: { type: 'string' },
          embedding: { type: 'array', items: { type: 'number' } },
          limit: { type: 'integer', minimum: 1, maximum: 50, default: 5 },
          time_query: { type: 'string' },
          start: { type: 'string' },
          end: { type: 'string' },
          tags: { type: 'array', items: { type: 'string' } },
          tag_mode: { type: 'string', enum: ['any', 'all'] },
          tag_match: { type: 'string', enum: ['exact', 'prefix'] }
        }
      }
    },
    {
      name: 'recall_memory_multi',
      description: 'Run multiple recall queries with server-side dedupe',
      inputSchema: {
        type: 'object',
        properties: {
          queries: { type: 'array', items: { type: 'string' } },
          limit_per_query: { type: 'integer', minimum: 1, maximum: 50, default: 5 },
          overall_limit: { type: 'integer', minimum: 1, maximum: 200, default: 15 },
          time_query: { type: 'string' },
          start: { type: 'string' },
          end: { type: 'string' },
          tags: { type: 'array', items: { type: 'string' } },
          tag_mode: { type: 'string', enum: ['any', 'all'] },
          tag_match: { type: 'string', enum: ['exact', 'prefix'] }
        },
        required: ['queries']
      }
    },
    {
      name: 'associate_memories',
      description: 'Create an association between two memories with a relationship type and strength',
      inputSchema: {
        type: 'object',
        properties: {
          memory1_id: { type: 'string' },
          memory2_id: { type: 'string' },
          type: { type: 'string' },
          strength: { type: 'number', minimum: 0, maximum: 1 }
        },
        required: ['memory1_id', 'memory2_id', 'type', 'strength']
      }
    },
    {
      name: 'update_memory',
      description: 'Update an existing memory (content, tags, metadata, timestamps, importance)',
      inputSchema: {
        type: 'object',
        properties: {
          memory_id: { type: 'string' },
          content: { type: 'string' },
          tags: { type: 'array', items: { type: 'string' } },
          importance: { type: 'number', minimum: 0, maximum: 1 },
          embedding: { type: 'array', items: { type: 'number' } },
          metadata: { type: 'object' },
          timestamp: { type: 'string' },
          updated_at: { type: 'string' },
          last_accessed: { type: 'string' }
        },
        required: ['memory_id']
      }
    },
    {
      name: 'delete_memory',
      description: 'Delete a memory by ID',
      inputSchema: {
        type: 'object',
        properties: { memory_id: { type: 'string' } },
        required: ['memory_id']
      }
    },
    {
      name: 'check_database_health',
      description: 'Check AutoMem service health',
      inputSchema: { type: 'object', properties: {} }
    }
  ];

  server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools }));

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    try {
      switch (name) {
        case 'store_memory': {
          const r = await client.storeMemory(args || {});
          return { content: [{ type: 'text', text: `Memory stored: ${r.memory_id}` }] };
        }
        case 'recall_memory': {
          const r = await client.recallMemory(args || {});
          const items = (r.results || r.memories || []).map((it, i) => {
            const mem = it.memory || it;
            const tags = mem.tags?.length ? ` [${mem.tags.join(', ')}]` : '';
            const score = it.final_score !== undefined ? ` score=${Number(it.final_score).toFixed(3)}` : '';
            return `${i + 1}. ${mem.content}${tags}${score}\n   ID: ${mem.id || mem.memory_id || ''}`;
          }).join('\n\n');
          const count = r.count ?? (r.results ? r.results.length : (r.memories ? r.memories.length : 0));
          return { content: [{ type: 'text', text: count ? `Found ${count} memories:\n\n${items}` : 'No memories found.' }] };
        }
        case 'recall_memory_multi': {
          const queries = Array.isArray(args?.queries) ? args.queries.filter(q => !!(q && q.trim())) : [];
          if (!queries.length) {
            throw new Error('queries is required and must include at least one non-empty query');
          }
          const perQueryLimit = Math.max(1, Math.min(args?.limit_per_query || args?.limit || 5, 50));
          const overallLimit = Math.max(1, Math.min(args?.overall_limit || args?.limit || queries.length * perQueryLimit || 15, 200));

          const r = await client.recallMemory({
            queries,
            limit: overallLimit,
            per_query_limit: perQueryLimit,
            time_query: args?.time_query,
            start: args?.start,
            end: args?.end,
            tags: Array.isArray(args?.tags) ? args.tags : undefined,
            tag_mode: args?.tag_mode,
            tag_match: args?.tag_match
          });

          const trimmed = r.results || r.memories || [];
          const items = trimmed.map((it, i) => {
            const mem = it.memory || it;
            const tags = mem.tags?.length ? ` [${mem.tags.join(', ')}]` : '';
            const score = it.final_score !== undefined ? ` score=${Number(it.final_score).toFixed(3)}` : '';
            const dedupNote = it.deduped_from?.length ? ` (deduped x${it.deduped_from.length})` : '';
            const queryNote = it._query ? ` query="${it._query}"` : '';
            return `${i + 1}. ${mem.content}${tags}${score}${dedupNote}${queryNote}\n   ID: ${mem.id || mem.memory_id || ''}`;
          }).join('\n\n');

          return {
            content: [{
              type: 'text',
              text: trimmed.length
                ? `Found ${trimmed.length} memories (removed ${r.dedup_removed ?? 0} duplicates) across ${queries.length} queries:\n\n${items}`
                : 'No memories found across the provided queries.'
            }]
          };
        }
        case 'associate_memories': {
          const r = await client.associateMemories(args || {});
          return { content: [{ type: 'text', text: r.message }] };
        }
        case 'update_memory': {
          const r = await client.updateMemory(args || {});
          return { content: [{ type: 'text', text: `Updated ${r.memory_id}` }] };
        }
        case 'delete_memory': {
          const r = await client.deleteMemory(args || {});
          return { content: [{ type: 'text', text: `Deleted ${r.memory_id}` }] };
        }
        case 'check_database_health': {
          const r = await client.checkHealth();
          return { content: [{ type: 'text', text: JSON.stringify(r) }] };
        }
        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    } catch (e) {
      return { content: [{ type: 'text', text: `Error: ${e.message || e}` }], isError: true };
    }
  });

  return server;
}

const app = express();
app.use(express.json({ limit: '4mb' }));

// Basic health
app.get('/health', (_req, res) => res.json({ status: 'healthy', mcp: 'sse', timestamp: new Date().toISOString() }));

// In-memory session store: sessionId -> { transport, server, res, heartbeat }
const sessions = new Map();

// Helper: validate and extract token from multiple sources
/**
 * Resolve the API token from request headers or query params.
 * Order: Bearer Authorization -> X-API-Key/X-API-Token header -> api_key/apiKey/api_token query.
 */
function getAuthToken(req) {
  const normalize = (v) => (typeof v === 'string' ? v.trim() : undefined);
  const auth = normalize(req.headers['authorization'] || '');
  const m = auth ? auth.match(/^Bearer\s+(.+)$/i) : null;
  const bearer = m ? normalize(m[1]) : undefined;
  const headerKey = normalize(req.headers['x-api-key'] || req.headers['x-api-token']);
  const queryKey = normalize(req.query.api_key || req.query.apiKey || req.query.api_token);
  return bearer || headerKey || queryKey;
}

// Alexa helpers
/**
 * Build a minimal Alexa speech response payload.
 */
function speech(text, { endSession = true } = {}) {
  return {
    version: '1.0',
    response: {
      outputSpeech: {
        type: 'PlainText',
        text,
      },
      shouldEndSession: endSession,
    },
  };
}

/**
 * Safely extract a slot value from an Alexa intent request.
 */
function getSlot(intent, name) {
  const slot = intent?.slots?.[name];
  return slot?.value || null;
}

/**
 * Construct tag set for Alexa requests including user and device context.
 */
function buildAlexaTags(body) {
  const tags = ['alexa'];
  const userId = body?.session?.user?.userId;
  const deviceId = body?.context?.System?.device?.deviceId;
  if (userId) tags.push(`user:${userId}`);
  if (deviceId) tags.push(`device:${deviceId}`);
  return tags;
}

/**
 * Convert recall results into short Alexa-friendly speech text.
 */
function formatRecallSpeech(records, { limit = 2 } = {}) {
  const items = (records || []).slice(0, limit).map((r, idx) => {
    const mem = r.memory || r;
    const text = typeof mem === 'string' ? mem : mem?.content || mem?.text || '';
    const trimmed = String(text || '').trim();
    const shortened = trimmed.length > 240 ? `${trimmed.slice(0, 240)}...` : trimmed;
    return `Item ${idx + 1}: ${shortened || 'empty'}`;
  });
  return items.length ? items.join(' ') : 'I could not find anything in memory for that.';
}

// Alexa skill endpoint (remember/recall via AutoMem)
app.post('/alexa', async (req, res) => {
  const body = req.body || {};
  const endpoint =
    body?.endpoint ||
    req.query.endpoint ||
    process.env.AUTOMEM_ENDPOINT ||
    'http://127.0.0.1:8001';
  const apiKey = getAuthToken(req) || process.env.AUTOMEM_API_TOKEN;

  if (!endpoint || !apiKey) {
    return res.status(500).json({ error: 'AutoMem endpoint or token not configured' });
  }

  const intentType = body?.request?.type;
  if (intentType === 'LaunchRequest') {
    return res.json(speech('AutoMem is ready. Say remember to store something, or recall to fetch it.', { endSession: false }));
  }
  if (intentType !== 'IntentRequest') {
    return res.json(speech('I did not understand that request.'));
  }

  const intent = body.request.intent;
  const name = intent?.name;
  const tags = buildAlexaTags(body);
  const client = new AutoMemClient({ endpoint, apiKey });

  if (name === 'RememberIntent') {
    const note = getSlot(intent, 'note');
    if (!note) {
      return res.json(speech('I did not hear anything to remember.', { endSession: false }));
    }
    try {
      await client.storeMemory({ content: note, tags });
      return res.json(speech('Saved to memory.', { endSession: false }));
    } catch (error) {
      console.error('[Alexa] storeMemory failed:', error.message);
      return res.json(speech('I could not save that right now.', { endSession: false }));
    }
  }

  if (name === 'RecallIntent') {
    const query = getSlot(intent, 'query');
    if (!query) {
      return res.json(speech('What should I recall?', { endSession: false }));
    }
    try {
      // First try scoped tags; if nothing, fall back to untagged recall
      const primary = await client.recallMemory({ query, tags, limit: 5 });
      const recordsPrimary = Array.isArray(primary?.results)
        ? primary.results
        : Array.isArray(primary?.memories)
          ? primary.memories
          : [];
      if (recordsPrimary.length) {
        const reply = formatRecallSpeech(recordsPrimary, { limit: 3 });
        return res.json(speech(reply, { endSession: false }));
      }

      const fallback = await client.recallMemory({ query, limit: 5 });
      const recordsFallback = Array.isArray(fallback?.results)
        ? fallback.results
        : Array.isArray(fallback?.memories)
          ? fallback.memories
          : [];
      const reply = formatRecallSpeech(recordsFallback, { limit: 3 });
      return res.json(speech(reply, { endSession: false }));
    } catch (error) {
      console.error('[Alexa] recallMemory failed:', error.message);
      return res.json(speech('I could not recall anything right now.', { endSession: false }));
    }
  }

  if (name === 'AMAZON.HelpIntent') {
    return res.json(speech('Say remember and a note to store it. Say recall and a topic to fetch it.', { endSession: false }));
  }

  return res.json(speech("I'm not sure how to handle that intent.", { endSession: false }));
});

// SSE endpoint
app.get('/mcp/sse', async (req, res) => {
  try {
    const endpoint = process.env.AUTOMEM_ENDPOINT || 'http://127.0.0.1:8001';
    const token = getAuthToken(req) || process.env.AUTOMEM_API_TOKEN;
    if (!endpoint) return res.status(500).json({ error: 'AUTOMEM_ENDPOINT not configured' });
    if (!token) return res.status(401).json({ error: 'Missing API token (use Authorization: Bearer, X-API-Key, or ?api_key=)' });

    const client = new AutoMemClient({ endpoint, apiKey: token });
    const server = buildMcpServer(client);
    // Help with proxy buffering before SSE headers are written
    res.set('X-Accel-Buffering', 'no');
    res.set('Cache-Control', 'no-cache, no-transform');
    const transport = new SSEServerTransport('/mcp/messages', res);
    await server.connect(transport);

    // Prepare session and lifecycle BEFORE sending the endpoint event to avoid race
    const heartbeat = setInterval(() => {
      try { res.write(': ping\n\n'); } catch (_) { /* ignore */ }
    }, 20000);
    res.on('close', () => {
      clearInterval(heartbeat);
      sessions.delete(transport.sessionId);
    });
    sessions.set(transport.sessionId, { transport, server, res, heartbeat });
    console.log(`[MCP] New SSE session established: ${transport.sessionId}`);

    // Now start SSE (writes event: endpoint)
    await transport.start();
  } catch (e) {
    try { res.status(500).json({ error: String(e) }); } catch (_) { /* ignore */ }
  }
});

// Message POST endpoint
app.post('/mcp/messages', async (req, res) => {
  const sessionId = req.query.sessionId;
  if (!sessionId || typeof sessionId !== 'string') return res.status(400).send('Missing sessionId');
  const s = sessions.get(sessionId);
  if (!s) {
    console.warn(`[MCP] POST for unknown session: ${sessionId}`);
    return res.status(404).send('Session not found');
  }
  try {
    await s.transport.handlePostMessage(req, res, req.body);
  } catch (e) {
    console.error(`[MCP] Error handling message for session ${sessionId}:`, e);
    try { res.status(400).send(String(e)); } catch (_) { /* ignore */ }
  }
});

const port = process.env.PORT || 8080;
app.listen(port, () => {
  console.log(`AutoMem MCP SSE server listening on :${port}`);
});
