// MCP server that bridges to AutoMem HTTP API
// Supports both Streamable HTTP (2025-03-26) and deprecated SSE (2024-11-05) transports
// Exposes:
//   ALL  /mcp           -> Streamable HTTP (POST to init, GET/POST/DELETE with Mcp-Session-Id)
//   GET  /mcp/sse       -> SSE stream (deprecated, clients POST JSON-RPC to /mcp/messages)
//   POST /mcp/messages  -> Accepts JSON-RPC messages for SSE sessions
//   GET  /health        -> Health probe

import express from 'express';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';

const DEFAULT_UPSTREAM_TIMEOUT_MS = 15000;
const DEFAULT_UPSTREAM_MAX_RETRIES = 2;
const DEFAULT_HEALTH_TIMEOUT_MS = 5000;
const DEFAULT_HEALTH_PROBE_INTERVAL_MS = 30000;
const TRANSIENT_STATUS_CODES = new Set([408, 429, 502, 503, 504]);

function readIntEnv(name, fallback) {
  const raw = process.env[name];
  if (!raw) return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function log(level, msg, extra = {}) {
  const line = JSON.stringify({ ts: new Date().toISOString(), level, msg, ...extra });
  if (level === 'error') {
    console.error(line);
  } else if (level === 'warn') {
    console.warn(line);
  } else {
    console.log(line);
  }
}

async function parseResponseBody(res) {
  const contentType = res.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    try {
      return await res.json();
    } catch (_) {
      return {};
    }
  }

  const text = await res.text();
  return text ? { message: text } : {};
}

function summarizeUpstreamErrorBody(status, data) {
  const message = data?.message || data?.detail || data?.error;
  return message ? String(message) : `HTTP ${status}`;
}

function isRetryableFetchError(error) {
  if (!error) return false;
  const name = error.name || '';
  return name === 'AbortError' || name === 'TimeoutError' || error instanceof TypeError;
}

class UpstreamRequestError extends Error {
  constructor(message, { status, requestId, kind, retryable = false, endpoint, cause } = {}) {
    super(message);
    this.name = 'UpstreamRequestError';
    this.status = status;
    this.requestId = requestId;
    this.kind = kind || 'upstream';
    this.retryable = retryable;
    this.endpoint = endpoint;
    this.cause = cause;
  }
}

function formatToolError(error, requestId) {
  const suffix = requestId ? ` (request_id: ${requestId})` : '';
  if (error instanceof UpstreamRequestError) {
    if (error.kind === 'timeout') {
      return `AutoMem request timed out. The service may be slow or restarting.${suffix}`;
    }
    if (error.status && TRANSIENT_STATUS_CODES.has(error.status)) {
      return `AutoMem service is temporarily unavailable (${error.message}). Please retry.${suffix}`;
    }
    return `AutoMem error: ${error.message}${suffix}`;
  }

  return `AutoMem error: ${error?.message || error}${suffix}`;
}

function sanitizeUrlForLog(rawUrl) {
  try {
    const parsed = new URL(rawUrl);
    return `${parsed.origin}${parsed.pathname}`;
  } catch {
    return rawUrl.split('?')[0];
  }
}

async function fetchWithRetry(url, { method, headers, body, requestId, timeoutMs, maxRetries } = {}) {
  const retries = Math.max(0, maxRetries ?? DEFAULT_UPSTREAM_MAX_RETRIES);
  const logUrl = sanitizeUrlForLog(url);

  for (let attempt = 0; attempt <= retries; attempt += 1) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const res = await fetch(url, { method, headers, body, signal: controller.signal });
      const data = await parseResponseBody(res);

      if (res.ok) {
        if (attempt > 0) {
          log('info', 'upstream_request_recovered', {
            reqId: requestId,
            url: logUrl,
            method,
            attempt: attempt + 1,
            status: res.status,
          });
        }
        return data;
      }

      const message = summarizeUpstreamErrorBody(res.status, data);
      const retryable = TRANSIENT_STATUS_CODES.has(res.status);
      log(retryable && attempt < retries ? 'warn' : 'error', 'upstream_http_error', {
        reqId: requestId,
        url: logUrl,
        method,
        attempt: attempt + 1,
        status: res.status,
        retryable,
        message,
      });

      if (retryable && attempt < retries) {
        await sleep(250 * (2 ** attempt) + Math.floor(Math.random() * 100));
        continue;
      }

      throw new UpstreamRequestError(message, {
        status: res.status,
        requestId,
        kind: 'http',
        retryable,
        endpoint: url,
      });
    } catch (error) {
      if (error instanceof UpstreamRequestError) {
        throw error;
      }

      const aborted = error?.name === 'AbortError';
      const retryable = isRetryableFetchError(error);
      if (retryable && attempt < retries) {
        log('warn', aborted ? 'upstream_timeout_retry' : 'upstream_fetch_retry', {
          reqId: requestId,
          url: logUrl,
          method,
          attempt: attempt + 1,
          timeoutMs,
          error: error?.message || String(error),
        });
        await sleep(250 * (2 ** attempt) + Math.floor(Math.random() * 100));
        continue;
      }

      log('error', aborted ? 'upstream_timeout' : 'upstream_fetch_failed', {
        reqId: requestId,
        url: logUrl,
        method,
        attempt: attempt + 1,
        timeoutMs,
        error: error?.message || String(error),
      });

      throw new UpstreamRequestError(
        aborted ? `request timed out after ${timeoutMs}ms` : `fetch failed: ${error?.message || error}`,
        {
          requestId,
          kind: aborted ? 'timeout' : 'network',
          retryable,
          endpoint: url,
          cause: error,
        }
      );
    } finally {
      clearTimeout(timer);
    }
  }
}

// Event store for resumable streams (Last-Event-ID support)
class InMemoryEventStore {
  constructor({ ttlMs = 60 * 60 * 1000, sweepMs = 5 * 60 * 1000 } = {}) {
    this.events = new Map(); // streamId -> { lastAccess, events: [{eventId, message}] }
    this.ttlMs = ttlMs;
    this.cleanupTimer = setInterval(() => {
      const now = Date.now();
      for (const [streamId, data] of this.events.entries()) {
        if (now - data.lastAccess > this.ttlMs) {
          this.events.delete(streamId);
        }
      }
    }, sweepMs);
    this.cleanupTimer.unref?.();
  }
  stopCleanup() {
    if (this.cleanupTimer) clearInterval(this.cleanupTimer);
  }
  removeStream(streamId) {
    this.events.delete(streamId);
  }
  async storeEvent(streamId, message) {
    const eventId = `${streamId}-${Date.now()}-${randomUUID().slice(0, 8)}`;
    if (!this.events.has(streamId)) {
      this.events.set(streamId, { lastAccess: Date.now(), events: [] });
    }
    const data = this.events.get(streamId);
    data.lastAccess = Date.now();
    data.events.push({ eventId, message });
    // Keep max 1000 events per stream
    if (data.events.length > 1000) data.events.shift();
    return eventId;
  }
  async replayEventsAfter(streamId, lastEventId) {
    const data = this.events.get(streamId);
    if (data) data.lastAccess = Date.now();
    const events = data?.events || [];
    const idx = events.findIndex(e => e.eventId === lastEventId);
    return idx >= 0 ? events.slice(idx + 1).map(e => e.message) : [];
  }
}

// Simple AutoMem HTTP client (mirrors the npm package behavior but inline to avoid version conflicts)
export class AutoMemClient {
  constructor(config) {
    this.config = config;
  }
  async _request(method, path, body, options = {}) {
    const url = `${this.config.endpoint.replace(/\/$/, '')}/${path.replace(/^\//, '')}`;
    const headers = { 'Content-Type': 'application/json' };
    if (this.config.apiKey) headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    const requestId = options.requestId || randomUUID();
    const timeoutMs = options.timeoutMs ?? readIntEnv('UPSTREAM_TIMEOUT_MS', DEFAULT_UPSTREAM_TIMEOUT_MS);
    const maxRetries = options.maxRetries ?? readIntEnv('UPSTREAM_MAX_RETRIES', DEFAULT_UPSTREAM_MAX_RETRIES);

    log('info', 'upstream_request', { reqId: requestId, method, url: sanitizeUrlForLog(url), timeoutMs, maxRetries });
    return fetchWithRetry(url, {
      method,
      headers,
      body: method === 'GET' ? undefined : (body ? JSON.stringify(body) : undefined),
      requestId,
      timeoutMs,
      maxRetries,
    });
  }
  async storeMemory(args, options) {
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
    const r = await this._request('POST', 'memory', body, options);
    return { memory_id: r.memory_id || r.id, message: r.message || 'Memory stored successfully' };
  }
  async recallMemory(args, options) {
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
    const r = await this._request('GET', path, undefined, options);
    return r;
  }
  async associateMemories(args, options) {
    const r = await this._request('POST', 'associate', {
      memory1_id: args.memory1_id,
      memory2_id: args.memory2_id,
      type: args.type,
      strength: args.strength
    }, options);
    return { success: true, message: r.message || 'Association created successfully' };
  }
  async updateMemory(args, options) {
    const { memory_id, ...updates } = args;
    const r = await this._request('PATCH', `memory/${memory_id}`, updates, options);
    return { memory_id: r.memory_id || memory_id, message: r.message || 'Memory updated successfully' };
  }
  async deleteMemory(args, options) {
    const r = await this._request('DELETE', `memory/${args.memory_id}`, undefined, options);
    return { memory_id: r.memory_id || args.memory_id, message: r.message || 'Memory deleted successfully' };
  }
  async checkHealth(options = {}) {
    return this._request('GET', 'health', undefined, {
      requestId: options.requestId,
      timeoutMs: options.timeoutMs ?? readIntEnv('HEALTH_TIMEOUT_MS', DEFAULT_HEALTH_TIMEOUT_MS),
      maxRetries: options.maxRetries ?? 0,
    });
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

  // Authorable relationship types must stay in sync with automem/config.py AUTHORABLE_RELATIONS
  const RELATION_TYPES = [
    'RELATES_TO', 'LEADS_TO', 'OCCURRED_BEFORE',
    'PREFERS_OVER', 'EXEMPLIFIES', 'CONTRADICTS', 'REINFORCES', 'INVALIDATED_BY',
    'EVOLVED_INTO', 'DERIVED_FROM', 'PART_OF',
  ];

  const MEMORY_TYPES = ['Decision', 'Pattern', 'Preference', 'Style', 'Habit', 'Insight', 'Context'];

  const tools = [
    {
      name: 'store_memory',
      description: 'Store a memory with optional tags, importance, metadata, timestamps, and embedding',
      annotations: { readOnlyHint: false, destructiveHint: false },
      inputSchema: {
        type: 'object',
        properties: {
          content: { type: 'string', description: 'Memory content text' },
          type: { type: 'string', enum: MEMORY_TYPES, description: 'Memory type for classification' },
          confidence: { type: 'number', minimum: 0, maximum: 1, description: 'Classification confidence (0-1, default 0.9 when type provided)' },
          tags: { type: 'array', items: { type: 'string' }, description: 'Tags for categorization and filtering' },
          importance: { type: 'number', minimum: 0, maximum: 1, description: 'Importance score (0-1, default 0.5)' },
          metadata: { type: 'object', description: 'Arbitrary key-value metadata' },
          timestamp: { type: 'string', description: 'ISO 8601 creation timestamp (defaults to now)' },
          id: { type: 'string', description: 'Custom memory ID (auto-generated if omitted)' },
          t_valid: { type: 'string', description: 'ISO 8601 timestamp when the memory becomes valid' },
          t_invalid: { type: 'string', description: 'ISO 8601 timestamp when the memory expires' },
          embedding: { type: 'array', items: { type: 'number' }, description: 'Pre-computed embedding vector (auto-generated if omitted)' },
          updated_at: { type: 'string', description: 'ISO 8601 last-updated timestamp' },
          last_accessed: { type: 'string', description: 'ISO 8601 last-accessed timestamp' },
        },
        required: ['content']
      }
    },
    {
      name: 'recall_memory',
      description: 'Recall memories with hybrid semantic/keyword search and optional time/tag filters',
      annotations: { readOnlyHint: true, destructiveHint: false },
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
            description: 'Result ordering (use time_* for chronological recaps)',
          },
          tags: { type: 'array', items: { type: 'string' }, description: 'Filter by tags' },
          tag_mode: { type: 'string', enum: ['any', 'all'], description: 'How to combine multiple tags (default: any)' },
          tag_match: { type: 'string', enum: ['exact', 'prefix'], description: 'How to match tags (default: exact)' },

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
      annotations: { readOnlyHint: false, destructiveHint: false },
      inputSchema: {
        type: 'object',
        properties: {
          memory1_id: { type: 'string', description: 'ID of the first memory (source)' },
          memory2_id: { type: 'string', description: 'ID of the second memory (target)' },
          type: {
            type: 'string',
            enum: RELATION_TYPES,
            description: 'Relationship type between the two memories',
          },
          strength: { type: 'number', minimum: 0, maximum: 1, description: 'Relationship strength (0-1)' },
        },
        required: ['memory1_id', 'memory2_id', 'type', 'strength']
      }
    },
    {
      name: 'update_memory',
      description: 'Update an existing memory (content, tags, metadata, timestamps, importance, type, confidence)',
      annotations: { readOnlyHint: false, destructiveHint: false },
      inputSchema: {
        type: 'object',
        properties: {
          memory_id: { type: 'string', description: 'ID of the memory to update' },
          content: { type: 'string', description: 'Updated memory content' },
          type: { type: 'string', enum: MEMORY_TYPES, description: 'Updated memory type' },
          confidence: { type: 'number', minimum: 0, maximum: 1, description: 'Updated classification confidence (0-1)' },
          tags: { type: 'array', items: { type: 'string' }, description: 'Updated tags (replaces existing)' },
          importance: { type: 'number', minimum: 0, maximum: 1, description: 'Updated importance score (0-1)' },
          metadata: { type: 'object', description: 'Updated metadata (merged with existing)' },
          timestamp: { type: 'string', description: 'Updated ISO 8601 creation timestamp' },
          embedding: { type: 'array', items: { type: 'number' }, description: 'Updated embedding vector' },
          updated_at: { type: 'string', description: 'ISO 8601 last-updated timestamp' },
          last_accessed: { type: 'string', description: 'ISO 8601 last-accessed timestamp' },
        },
        required: ['memory_id']
      }
    },
    {
      name: 'delete_memory',
      description: 'Delete a memory by ID',
      annotations: { readOnlyHint: false, destructiveHint: true },
      inputSchema: {
        type: 'object',
        properties: {
          memory_id: { type: 'string', description: 'ID of the memory to delete' },
        },
        required: ['memory_id']
      }
    },
    {
      name: 'check_database_health',
      description: 'Check AutoMem service health (FalkorDB, Qdrant, embedding provider)',
      annotations: { readOnlyHint: true, destructiveHint: false },
      inputSchema: { type: 'object', properties: {} }
    }
  ];

  server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools }));

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    const requestId = randomUUID();
    try {
      switch (name) {
        case 'store_memory': {
          const r = await client.storeMemory(args || {}, { requestId });
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

          const r = await client.recallMemory(recallArgs, { requestId });
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
          const r = await client.associateMemories(args || {}, { requestId });
          return { content: [{ type: 'text', text: r.message }] };
        }
        case 'update_memory': {
          const r = await client.updateMemory(args || {}, { requestId });
          return { content: [{ type: 'text', text: `Updated ${r.memory_id}` }] };
        }
        case 'delete_memory': {
          const r = await client.deleteMemory(args || {}, { requestId });
          return { content: [{ type: 'text', text: `Deleted ${r.memory_id}` }] };
        }
        case 'check_database_health': {
          const r = await client.checkHealth({ requestId });
          return { content: [{ type: 'text', text: JSON.stringify(r) }] };
        }
        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    } catch (e) {
      log('error', 'tool_request_failed', {
        reqId: requestId,
        tool: name,
        error: e?.message || String(e),
        status: e?.status,
        kind: e?.kind,
      });
      return { content: [{ type: 'text', text: formatToolError(e, requestId) }], isError: true };
    }
  });

  return server;
}

export function createApp() {
  const app = express();
  app.use(express.json({ limit: '4mb' }));
  app.use((req, res, next) => {
    const requestId = typeof req.headers['x-request-id'] === 'string' && req.headers['x-request-id'].trim()
      ? req.headers['x-request-id'].trim()
      : randomUUID();
    req.requestId = requestId;
    res.set('X-Request-Id', requestId);
    next();
  });

  // Expose Mcp-Session-Id header for browser-based clients
  app.use((req, res, next) => {
    res.set('Access-Control-Expose-Headers', 'Mcp-Session-Id');
    next();
  });

  // In-memory session store for legacy SSE only.
  const sessions = new Map();

  // Sweep abandoned SSE sessions every 5 minutes (1-hour TTL)
  const SESSION_TTL_MS = 60 * 60 * 1000;
  const sessionSweep = setInterval(() => {
    const now = Date.now();
    for (const [sid, session] of sessions.entries()) {
      if (session.type === 'sse' && now - (session.lastAccess || 0) > SESSION_TTL_MS) {
        log('info', 'mcp_session_swept', { sessionId: sid, transport: 'sse' });
        // Delete first so transport.onclose guard (sessions.has) becomes a no-op
        sessions.delete(sid);
        // close() returns a Promise — catch async rejections to avoid unhandled rejection crashes
        if (session.transport) {
          Promise.resolve(session.transport.close()).catch(() => {});
        }
        if (session.server) {
          Promise.resolve(session.server.close()).catch(() => {});
        }
        session.eventStore?.removeStream(sid);
        session.eventStore?.stopCleanup();
      }
    }
  }, 5 * 60 * 1000);
  sessionSweep.unref?.();

  const healthState = {
    checked_at: null,
    status: 'starting',
    upstream: 'unknown',
    details: null,
    error: null,
  };
  let healthProbePromise = null;

  async function probeUpstreamHealth(trigger = 'request') {
    if (healthProbePromise) return healthProbePromise;

    healthProbePromise = (async () => {
      const endpoint = process.env.AUTOMEM_API_URL || process.env.AUTOMEM_ENDPOINT || 'http://127.0.0.1:8001';
      const token = process.env.AUTOMEM_API_TOKEN;
      const requestId = `health-${randomUUID()}`;

      if (!token) {
        healthState.checked_at = new Date().toISOString();
        healthState.status = 'degraded';
        healthState.upstream = 'unconfigured';
        healthState.details = null;
        healthState.error = 'AUTOMEM_API_TOKEN not configured';
        log('warn', 'health_probe_unconfigured', { reqId: requestId, trigger });
        return healthState;
      }

      try {
        const client = new AutoMemClient({ endpoint, apiKey: token });
        const result = await client.checkHealth({
          requestId,
          timeoutMs: readIntEnv('HEALTH_TIMEOUT_MS', DEFAULT_HEALTH_TIMEOUT_MS),
          maxRetries: 0,
        });

        healthState.checked_at = new Date().toISOString();
        healthState.status = result?.status === 'healthy' ? 'healthy' : 'degraded';
        healthState.upstream = 'reachable';
        healthState.details = result;
        healthState.error = null;
        log('info', 'health_probe_ok', {
          reqId: requestId,
          trigger,
          upstreamStatus: result?.status || 'unknown',
        });
      } catch (error) {
        healthState.checked_at = new Date().toISOString();
        healthState.status = 'degraded';
        healthState.upstream = 'unreachable';
        healthState.details = null;
        healthState.error = error?.message || String(error);
        log('warn', 'health_probe_failed', { reqId: requestId, trigger, error: healthState.error });
      }

      return healthState;
    })();

    try {
      return await healthProbePromise;
    } finally {
      healthProbePromise = null;
    }
  }

  const healthProbeTimer = setInterval(() => {
    void probeUpstreamHealth('interval');
  }, readIntEnv('HEALTH_PROBE_INTERVAL_MS', DEFAULT_HEALTH_PROBE_INTERVAL_MS));
  healthProbeTimer.unref?.();
  void probeUpstreamHealth('startup');

  app.get('/health', async (req, res) => {
    if (!healthState.checked_at) {
      await probeUpstreamHealth('request');
    }

    const body = {
      status: healthState.status,
      transports: ['streamable-http', 'sse'],
      endpoints: { streamableHttp: '/mcp', sse: '/mcp/sse' },
      upstream: healthState.upstream,
      upstream_error: healthState.error,
      upstream_details: healthState.details,
      checked_at: healthState.checked_at,
      timestamp: new Date().toISOString(),
      request_id: req.requestId,
    };

    res.status(healthState.status === 'healthy' ? 200 : 503).json(body);
  });

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
    process.env.AUTOMEM_API_URL ||
    process.env.AUTOMEM_ENDPOINT ||  // Legacy fallback
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
      await client.storeMemory({ content: note, tags }, { requestId: req.requestId });
      return res.json(speech('Saved to memory.', { endSession: false }));
    } catch (error) {
      log('error', 'alexa_store_failed', { reqId: req.requestId, error: error?.message || String(error) });
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
      const primary = await client.recallMemory({ query, tags, limit: 5 }, { requestId: req.requestId });
      const recordsPrimary = Array.isArray(primary?.results)
        ? primary.results
        : Array.isArray(primary?.memories)
          ? primary.memories
          : [];
      if (recordsPrimary.length) {
        const reply = formatRecallSpeech(recordsPrimary, { limit: 3 });
        return res.json(speech(reply, { endSession: false }));
      }

      const fallback = await client.recallMemory({ query, limit: 5 }, { requestId: req.requestId });
      const recordsFallback = Array.isArray(fallback?.results)
        ? fallback.results
        : Array.isArray(fallback?.memories)
          ? fallback.memories
          : [];
      const reply = formatRecallSpeech(recordsFallback, { limit: 3 });
      return res.json(speech(reply, { endSession: false }));
    } catch (error) {
      log('error', 'alexa_recall_failed', { reqId: req.requestId, error: error?.message || String(error) });
      return res.json(speech('I could not recall anything right now.', { endSession: false }));
    }
  }

  if (name === 'AMAZON.HelpIntent') {
    return res.json(speech('Say remember and a note to store it. Say recall and a topic to fetch it.', { endSession: false }));
  }

  return res.json(speech("I'm not sure how to handle that intent.", { endSession: false }));
  });

// Streamable HTTP endpoint (MCP 2025-03-26 protocol)
  app.all('/mcp', async (req, res) => {
    log('info', 'mcp_request', { reqId: req.requestId, method: req.method, path: req.path });
    res.set('X-Accel-Buffering', 'no');
    res.set('Cache-Control', 'no-cache, no-transform');

    try {
      const sessionId = req.headers['mcp-session-id'];
      if (sessionId) {
        log('info', 'mcp_session_header_ignored', {
          reqId: req.requestId,
          method: req.method,
          path: req.path,
          sessionId,
        });
      }

      const endpoint = process.env.AUTOMEM_API_URL || process.env.AUTOMEM_ENDPOINT || 'http://127.0.0.1:8001';
      const token = getAuthToken(req) || process.env.AUTOMEM_API_TOKEN;
      if (!token) {
        return res.status(401).json({ error: 'Missing API token (use Authorization: Bearer, X-API-Key, or ?api_key=)' });
      }

      const client = new AutoMemClient({ endpoint, apiKey: token });
      const server = buildMcpServer(client);
      const transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: undefined,
        enableJsonResponse: true,
      });

      res.on('close', () => {
        Promise.resolve(transport.close()).catch(() => {});
      });

      await server.connect(transport);
      await transport.handleRequest(req, res, req.body);
    } catch (e) {
      log('error', 'mcp_request_failed', {
        reqId: req.requestId,
        method: req.method,
        path: req.path,
        error: e?.message || String(e),
      });
      if (!res.headersSent) {
        res.status(500).json({
          jsonrpc: '2.0',
          error: { code: -32603, message: `Internal server error (request_id: ${req.requestId})` },
          id: null
        });
      }
    }
  });

// SSE endpoint (deprecated HTTP+SSE protocol 2024-11-05)
  app.get('/mcp/sse', async (req, res) => {
  try {
    const endpoint = process.env.AUTOMEM_API_URL || process.env.AUTOMEM_ENDPOINT || 'http://127.0.0.1:8001';
    const token = getAuthToken(req) || process.env.AUTOMEM_API_TOKEN;
    if (!endpoint) return res.status(500).json({ error: 'AUTOMEM_API_URL not configured' });
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
    sessions.set(transport.sessionId, { transport, server, res, heartbeat, type: 'sse' });
    log('info', 'mcp_session_initialized', { reqId: req.requestId, sessionId: transport.sessionId, transport: 'sse' });

    // Server.connect() starts the SSE transport in current MCP SDK versions.
  } catch (e) {
    log('error', 'mcp_sse_failed', { reqId: req.requestId, error: e?.message || String(e) });
    try { res.status(500).json({ error: String(e), request_id: req.requestId }); } catch (_) { /* ignore */ }
  }
  });

// Message POST endpoint
  app.post('/mcp/messages', async (req, res) => {
  const sessionId = req.query.sessionId;
  if (!sessionId || typeof sessionId !== 'string') return res.status(400).send('Missing sessionId');
  const s = sessions.get(sessionId);
  if (!s) {
    log('warn', 'mcp_unknown_session', { reqId: req.requestId, sessionId });
    return res.status(404).send('Session not found');
  }
  try {
    await s.transport.handlePostMessage(req, res, req.body);
  } catch (e) {
    log('error', 'mcp_message_failed', { reqId: req.requestId, sessionId, error: e?.message || String(e) });
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
    log('info', 'mcp_bridge_listening', { port });
  });
}
