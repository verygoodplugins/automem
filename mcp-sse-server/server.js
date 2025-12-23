// Minimal MCP-over-SSE server that bridges to AutoMem HTTP API
// Exposes:
//   GET  /mcp/sse       -> SSE stream (clients POST JSON-RPC to /mcp/messages)
//   POST /mcp/messages  -> Accepts JSON-RPC messages for active sessions
//   GET  /health        -> Health probe

import express from 'express';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { fileURLToPath } from 'node:url';

// Simple AutoMem HTTP client (mirrors the npm package behavior but inline to avoid version conflicts)
export class AutoMemClient {
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
    if (args.sort) p.set('sort', args.sort);
    if (Array.isArray(args.tags)) args.tags.forEach(t => p.append('tags', t));
    if (args.tag_mode) p.set('tag_mode', args.tag_mode);
    if (args.tag_match) p.set('tag_match', args.tag_match);

    // Advanced recall options (pass-through to AutoMem /recall)
    if (args.expand_relations !== undefined) p.set('expand_relations', String(!!args.expand_relations));
    if (args.expand_entities !== undefined) p.set('expand_entities', String(!!args.expand_entities));
    if (args.auto_decompose !== undefined) p.set('auto_decompose', String(!!args.auto_decompose));
    if (args.expansion_limit !== undefined) p.set('expansion_limit', String(args.expansion_limit));
    if (args.relation_limit !== undefined) p.set('relation_limit', String(args.relation_limit));
    if (args.expand_min_importance !== undefined) p.set('expand_min_importance', String(args.expand_min_importance));
    if (args.expand_min_strength !== undefined) p.set('expand_min_strength', String(args.expand_min_strength));

    if (args.context) p.set('context', args.context);
    if (args.language) p.set('language', args.language);
    if (args.active_path) p.set('active_path', args.active_path);
    if (Array.isArray(args.context_tags)) args.context_tags.forEach(t => { if (t) p.append('context_tags', t); });
    if (Array.isArray(args.context_types)) args.context_types.forEach(t => { if (t) p.append('context_types', t); });
    if (Array.isArray(args.priority_ids)) args.priority_ids.forEach(t => { if (t) p.append('priority_ids', t); });

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

export function formatRecallAsItems(results, { detailed = false } = {}) {
  return (results || []).map((it, i) => {
    const mem = it?.memory || it || {};
    const id = mem.id || mem.memory_id || it?.id || it?.memory_id || '';
    const content = mem.content ?? mem.text ?? '';

    const tags = Array.isArray(mem.tags) ? mem.tags.filter(t => typeof t === 'string' && t.trim()) : [];
    const score = it?.final_score !== undefined ? Number(it.final_score) : undefined;
    const dedupCount = Array.isArray(it?.deduped_from) ? it.deduped_from.length : 0;

    if (!detailed) {
      const tagSuffix = tags.length ? ` [${tags.join(', ')}]` : '';
      const scoreSuffix = score !== undefined ? ` score=${score.toFixed(3)}` : '';
      const dedupNote = dedupCount ? ` (deduped x${dedupCount})` : '';
      return {
        type: 'text',
        text: `${i + 1}. ${String(content)}${tagSuffix}${scoreSuffix}${dedupNote}\nID: ${id}`,
      };
    }

    const lines = [];
    lines.push(`${i + 1}. ${String(content)}`);
    if (id) lines.push(`ID: ${id}`);
    if (mem.type) lines.push(`Type: ${String(mem.type)}`);
    if (mem.timestamp) lines.push(`Timestamp: ${String(mem.timestamp)}`);
    if (mem.last_accessed) lines.push(`Last accessed: ${String(mem.last_accessed)}`);
    if (mem.importance !== undefined) {
      const imp = Number(mem.importance);
      lines.push(`Importance: ${Number.isFinite(imp) ? imp.toFixed(3) : String(mem.importance)}`);
    }
    if (mem.confidence !== undefined) {
      const conf = Number(mem.confidence);
      lines.push(`Confidence: ${Number.isFinite(conf) ? conf.toFixed(3) : String(mem.confidence)}`);
    }
    if (tags.length) lines.push(`Tags: ${tags.join(', ')}`);
    if (score !== undefined) lines.push(`Score: ${score.toFixed(3)}`);
    if (it?.match_type) lines.push(`Match: ${String(it.match_type)}`);
    if (it?.source) lines.push(`Source: ${String(it.source)}`);

    // Associations (only present on relation-expanded results)
    const rels = Array.isArray(it?.relations) ? it.relations : [];
    if (rels.length) {
      const summarized = rels.slice(0, 5).map(r => {
        const t = r?.type ? String(r.type) : 'REL';
        const s = r?.strength !== undefined ? Number(r.strength) : undefined;
        const from = r?.from ? String(r.from) : '';
        const ss = s !== undefined && Number.isFinite(s) ? `(${s.toFixed(2)})` : '';
        return `${t}${ss}${from ? ` from ${from}` : ''}`;
      });
      lines.push(`Relations: ${summarized.join('; ')}${rels.length > 5 ? ` (+${rels.length - 5} more)` : ''}`);
    }

    return { type: 'text', text: lines.join('\n') };
  });
}

// Build a new MCP Server instance with AutoMem tool handlers
export function buildMcpServer(client) {
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
          query: { type: 'string', description: 'Text query to search for in memory content' },
          queries: { type: 'array', items: { type: 'string' }, description: 'Multiple queries (server-side deduplication)' },
          embedding: { type: 'array', items: { type: 'number' }, description: 'Embedding vector for semantic similarity search' },
          limit: { type: 'integer', minimum: 1, maximum: 50, default: 5, description: 'Maximum number of memories to return' },
          per_query_limit: { type: 'integer', minimum: 1, maximum: 50, description: 'Per-query limit when using queries[]' },
          time_query: { type: 'string', description: 'Natural language time window (e.g. "today", "last week")' },
          start: { type: 'string', description: 'Explicit ISO timestamp lower bound' },
          end: { type: 'string', description: 'Explicit ISO timestamp upper bound' },
          sort: {
            type: 'string',
            enum: ['score', 'time_desc', 'time_asc', 'updated_desc', 'updated_asc'],
            description: 'Result ordering (use time_* for chronological recaps).',
          },
          tags: { type: 'array', items: { type: 'string' }, description: 'Filter by tags' },
          tag_mode: { type: 'string', enum: ['any', 'all'], description: 'How to combine multiple tags' },
          tag_match: { type: 'string', enum: ['exact', 'prefix'], description: 'How to match tags' },

          // Advanced AutoMem /recall options
          expand_relations: { type: 'boolean', description: 'Enable graph relation expansion' },
          expand_entities: { type: 'boolean', description: 'Enable entity-based multi-hop expansion' },
          auto_decompose: { type: 'boolean', description: 'Auto-generate supplementary queries from query text' },
          expansion_limit: { type: 'integer', minimum: 1, maximum: 500, description: 'Max number of expanded results' },
          relation_limit: { type: 'integer', minimum: 1, maximum: 200, description: 'Max relations to expand per seed memory' },
          expand_min_importance: { type: 'number', minimum: 0, maximum: 1, description: 'Filter expanded results by min importance' },
          expand_min_strength: { type: 'number', minimum: 0, maximum: 1, description: 'Filter expanded results by min relation strength' },
          context: { type: 'string', description: 'Context label (e.g. coding-style, architecture)' },
          language: { type: 'string', description: 'Programming language hint (e.g. python, typescript)' },
          active_path: { type: 'string', description: 'Active file path hint (e.g. src/auth.ts)' },
          context_tags: { type: 'array', items: { type: 'string' }, description: 'Priority tags to boost in results' },
          context_types: { type: 'array', items: { type: 'string' }, description: 'Priority memory types to boost (Decision, Pattern, ...)' },
          priority_ids: { type: 'array', items: { type: 'string' }, description: 'Specific memory IDs to include/boost' },

          format: {
            type: 'string',
            enum: ['text', 'items', 'detailed', 'json'],
            default: 'text',
            description: 'Output formatting: text (single block), items (one memory per content item), detailed (per-item with timestamps/relations), json (raw response JSON as text)',
          }
        }
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
          // Unified handler: supports single query OR multiple queries
          const queries = Array.isArray(args?.queries) ? args.queries.filter(q => !!(q && q.trim())) : [];
          const isMulti = queries.length > 0;

          const recallArgs = {
            query: args?.query,
            queries: isMulti ? queries : undefined,
            limit: args?.limit || (isMulti ? queries.length * 5 : 5),
            per_query_limit: isMulti ? Math.min(args?.per_query_limit || args?.limit || 5, 50) : undefined,
            embedding: args?.embedding,
            time_query: args?.time_query,
            start: args?.start,
            end: args?.end,
            sort: args?.sort,
            tags: Array.isArray(args?.tags) ? args.tags : undefined,
            tag_mode: args?.tag_mode,
            tag_match: args?.tag_match,

            expand_relations: args?.expand_relations,
            expand_entities: args?.expand_entities,
            auto_decompose: args?.auto_decompose,
            expansion_limit: args?.expansion_limit,
            relation_limit: args?.relation_limit,
            expand_min_importance: args?.expand_min_importance,
            expand_min_strength: args?.expand_min_strength,
            context: args?.context,
            language: args?.language,
            active_path: args?.active_path,
            context_tags: Array.isArray(args?.context_tags) ? args.context_tags : undefined,
            context_types: Array.isArray(args?.context_types) ? args.context_types : undefined,
            priority_ids: Array.isArray(args?.priority_ids) ? args.priority_ids : undefined,
          };

          const r = await client.recallMemory(recallArgs);
          const results = r.results || r.memories || [];

          const count = r.count ?? results.length;
          const dedupInfo = r.dedup_removed ? ` (${r.dedup_removed} duplicates removed)` : '';

          const format = (args?.format || 'text').toLowerCase();
          if (format === 'json') {
            return { content: [{ type: 'text', text: JSON.stringify(r, null, 2) }] };
          }

          if (format === 'items') {
            return {
              content: count
                ? [{ type: 'text', text: `Found ${count} memories${dedupInfo}:` }, ...formatRecallAsItems(results)]
                : [{ type: 'text', text: 'No memories found.' }],
            };
          }

          if (format === 'detailed') {
            return {
              content: count
                ? [{ type: 'text', text: `Found ${count} memories${dedupInfo}:` }, ...formatRecallAsItems(results, { detailed: true })]
                : [{ type: 'text', text: 'No memories found.' }],
            };
          }

          // Back-compat: preserve the old single-block text format as default.
          const itemsText = formatRecallAsItems(results).map(x => x.text.replace('\nID: ', '\n   ID: ')).join('\n\n');
          return {
            content: [{
              type: 'text',
              text: count ? `Found ${count} memories${dedupInfo}:\n\n${itemsText}` : 'No memories found.'
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

export function createApp() {
  const app = express();
  app.use(express.json({ limit: '4mb' }));

  // In-memory session store: sessionId -> { transport, server, res, heartbeat }
  const sessions = new Map();

  // Basic health
  app.get('/health', (_req, res) => res.json({ status: 'healthy', mcp: 'sse', timestamp: new Date().toISOString() }));

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

  return app;
}

const port = process.env.PORT || 8080;

// Avoid side effects on import (tests/tools may import this module).
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const app = createApp();
  app.listen(port, () => {
    console.log(`AutoMem MCP SSE server listening on :${port}`);
  });
}
