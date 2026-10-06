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

function stripUndefinedValues(obj) {
  return Object.fromEntries(Object.entries(obj).filter(([, value]) => value !== undefined));
}

const ASSOCIATION_FAILURE_DETAIL_LIMIT = 5;

function formatAssociationResultMessage(data) {
  if (!data?.summary) {
    return data?.message || 'Association created successfully';
  }

  const failures = Array.isArray(data.failed) ? data.failed : [];
  const visibleFailures = failures.slice(0, ASSOCIATION_FAILURE_DETAIL_LIMIT);
  const failureParts = visibleFailures
    .map((item) => `failed index ${item.index}: ${item.reason || 'unknown error'}`)
  const omittedCount = failures.length - visibleFailures.length;
  if (omittedCount > 0) {
    failureParts.push(`${omittedCount} more failure${omittedCount === 1 ? '' : 's'} omitted`);
  }
  return failureParts.length ? `${data.summary}; ${failureParts.join('; ')}` : data.summary;
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

// recall_memory is a port of @verygoodplugins/mcp-automem 0.16.0 so both
// transports serve one contract: the tool definition (src/mcp-surface.ts),
// mode routing and request mapping (src/automem-client.ts), and rendering
// (src/recall-memory.ts). test/recall-parity.test.js runs the stdio package's
// own code against the same canned API responses and fails on any difference.

// Matches the API's own rejection (automem/api/memory.py _validate_memory_id).
const INVALID_MEMORY_ID_MESSAGE = 'memory_id must be a valid UUID';
const CANONICAL_UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

// GET /memory/by-tag cannot honor these, so enumeration mode rejects them.
const RANKED_ONLY_RECALL_PARAMS = [
  'query',
  'queries',
  'embedding',
  'time_query',
  'start',
  'end',
  'exclude_tags',
  'expand_relations',
  'expand_entities',
  'auto_decompose',
  'expansion_limit',
  'relation_limit',
  'expand_min_importance',
  'expand_min_strength',
  'current_only',
  'state_debug',
  'state_mode',
  'recency_bias',
  'scope_fallback',
  'expand_respect_tags',
  'min_score',
  'adaptive_floor',
  'sort',
];

function nonEmptyTags(tags) {
  if (!Array.isArray(tags)) return [];
  return tags.map((t) => (typeof t === 'string' ? t.trim() : '')).filter((t) => t.length > 0);
}

function mapStoredMemory(raw) {
  return {
    memory_id: raw?.id || raw?.memory_id || '',
    content: raw?.content || '',
    summary: raw?.summary,
    tags: raw?.tags || [],
    importance: raw?.importance ?? 0,
    created_at: raw?.timestamp || raw?.created_at || '',
    updated_at: raw?.updated_at || raw?.timestamp || '',
    metadata: raw?.metadata || {},
    type: raw?.type,
    confidence: raw?.confidence,
    last_accessed: raw?.last_accessed,
  };
}

// ID fetch and enumeration return stored records, not scored hits; wrap them in
// the ranked result shape so one renderer serves all three modes.
function wrapMemoryAsRecallResult(raw) {
  const memory = mapStoredMemory(raw);
  return {
    id: memory.memory_id,
    match_type: 'direct',
    final_score: 1,
    score_components: {},
    relations: [],
    memory,
  };
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
  // Mode routing, validation and request mapping are a port of mcp-automem's
  // AutoMemClient.recallMemory (src/automem-client.ts). Unknown arguments used
  // to be dropped here, so an ID fetch fell through to an unfiltered ranked
  // search and returned unrelated memories without an error.
  async recallMemory(args = {}, options) {
    // Mode 1, ID fetch: GET /memory/{id}. Every other param is ignored.
    if (typeof args.memory_id === 'string' && args.memory_id.trim().length > 0) {
      const memory = await this.fetchMemoryById(args.memory_id.trim(), options);
      return {
        results: memory ? [wrapMemoryAsRecallResult(memory)] : [],
        count: memory ? 1 : 0,
        mode: 'id_fetch',
      };
    }

    // Mode 2, tag enumeration: GET /memory/by-tag, paginated exact-match listing.
    if (args.exhaustive === true) {
      const cleanTags = nonEmptyTags(args.tags);
      if (cleanTags.length === 0) {
        throw new Error('recall_memory: `exhaustive: true` requires non-empty `tags`');
      }
      if (args.tag_match && args.tag_match !== 'exact') {
        throw new Error(
          'recall_memory: enumeration mode (`exhaustive: true`) only supports exact tag matching; remove `tag_match: "prefix"`'
        );
      }
      if (args.tag_mode && args.tag_mode !== 'any') {
        throw new Error(
          'recall_memory: enumeration mode (`exhaustive: true`) only supports any-of tag matching; remove `tag_mode: "all"`'
        );
      }
      // A ranked-only param would silently change the meaning of the query:
      // `{ tags, exhaustive: true, time_query: "last 7 days" }` would list every
      // tagged memory and ignore the window.
      const conflicting = RANKED_ONLY_RECALL_PARAMS.filter((key) => {
        const v = args[key];
        if (v === undefined || v === null) return false;
        if (Array.isArray(v)) return v.length > 0;
        return true;
      });
      if (conflicting.length > 0) {
        throw new Error(
          `recall_memory: enumeration mode (\`exhaustive: true\`) only accepts \`tags\`, \`limit\`, \`offset\`, \`tag_mode: "any"\`, \`tag_match: "exact"\`, and \`format\`. Remove ranked-only param(s): ${conflicting.join(', ')}.`
        );
      }
      return this.listMemoriesByTag(cleanTags, args.limit, args.offset, options);
    }

    // Mode 3, ranked retrieval: GET /recall.
    const p = new URLSearchParams();
    if (args.query) p.set('query', args.query);
    if (Array.isArray(args.queries) && args.queries.length > 0) {
      args.queries.filter((q) => q && q.trim()).forEach((q) => p.append('queries', q));
    }
    if (args.limit) p.set('limit', String(args.limit));
    if (Array.isArray(args.embedding)) p.set('embedding', args.embedding.join(','));
    if (args.time_query) p.set('time_query', args.time_query);
    if (args.start) p.set('start', args.start);
    if (args.end) p.set('end', args.end);
    if (Array.isArray(args.tags)) args.tags.forEach((tag) => p.append('tags', tag));
    if (Array.isArray(args.exclude_tags) && args.exclude_tags.length > 0) {
      args.exclude_tags.forEach((tag) => p.append('exclude_tags', tag));
    }
    if (args.tag_mode === 'any' || args.tag_mode === 'all') p.set('tag_mode', args.tag_mode);
    if (args.tag_match === 'exact' || args.tag_match === 'prefix') p.set('tag_match', args.tag_match);

    // Graph expansion
    if (typeof args.expand_relations === 'boolean') p.set('expand_relations', String(args.expand_relations));
    if (typeof args.expand_respect_tags === 'boolean') {
      p.set('expand_respect_tags', String(args.expand_respect_tags));
    }
    if (typeof args.expand_entities === 'boolean') p.set('expand_entities', String(args.expand_entities));
    if (typeof args.auto_decompose === 'boolean') p.set('auto_decompose', String(args.auto_decompose));
    if (typeof args.expansion_limit === 'number') p.set('expansion_limit', String(args.expansion_limit));
    if (typeof args.relation_limit === 'number') p.set('relation_limit', String(args.relation_limit));
    if (typeof args.expand_min_importance === 'number') {
      p.set('expand_min_importance', String(args.expand_min_importance));
    }
    if (typeof args.expand_min_strength === 'number') {
      p.set('expand_min_strength', String(args.expand_min_strength));
    }

    // Current-state filtering, recency and score floors
    if (typeof args.current_only === 'boolean') p.set('current_only', String(args.current_only));
    if (typeof args.state_debug === 'boolean') p.set('state_debug', String(args.state_debug));
    if (args.state_mode === 'current' || args.state_mode === 'history') p.set('state_mode', args.state_mode);
    if (['auto', 'on', 'off'].includes(args.recency_bias)) p.set('recency_bias', args.recency_bias);
    if (typeof args.scope_fallback === 'boolean') p.set('scope_fallback', String(args.scope_fallback));
    if (typeof args.min_score === 'number') p.set('min_score', String(args.min_score));
    if (typeof args.adaptive_floor === 'boolean') p.set('adaptive_floor', String(args.adaptive_floor));

    // Context hints
    if (args.context) p.set('context', args.context);
    if (args.language) p.set('language', args.language);
    if (args.active_path) p.set('active_path', args.active_path);
    if (Array.isArray(args.context_tags) && args.context_tags.length > 0) {
      args.context_tags.forEach((tag) => p.append('context_tags', tag));
    }
    if (Array.isArray(args.context_types) && args.context_types.length > 0) {
      args.context_types.forEach((t) => p.append('context_types', t));
    }
    if (Array.isArray(args.priority_ids) && args.priority_ids.length > 0) {
      args.priority_ids.forEach((id) => p.append('priority_ids', id));
    }

    // Pagination and output control. /recall reads neither `format` nor
    // `offset` today; both are forwarded exactly as the stdio client does.
    if (args.per_query_limit !== undefined && args.per_query_limit > 0) {
      p.set('per_query_limit', String(args.per_query_limit));
    }
    if (args.sort) p.set('sort', args.sort);
    if (args.format) p.set('format', args.format);
    if (args.offset !== undefined && args.offset > 0) p.set('offset', String(args.offset));

    const path = p.toString() ? `recall?${p.toString()}` : 'recall';
    const response = await this._request('GET', path, undefined, options);
    return {
      results: (response.results || []).map((result) => ({
        id: result.id,
        match_type: result.match_type,
        match_score: result.match_score,
        relation_score: result.relation_score,
        final_score: result.final_score ?? result.score ?? 0,
        score_components: result.score_components || {},
        source: result.source,
        // Servers may send either key; normalize to one so the formatter never
        // serializes the same relation list twice.
        relations: result.relations || result.related_to || [],
        deduped_from: result.deduped_from,
        expanded_from_entity: result.expanded_from_entity,
        outside_tag_scope: result.outside_tag_scope,
        jit_enriched: result.jit_enriched,
        state_replaces: result.state_replaces,
        memory: {
          memory_id: result.id,
          content: result.memory?.content || '',
          summary: result.memory?.summary,
          tags: result.memory?.tags || [],
          importance: result.memory?.importance ?? 0,
          created_at: result.memory?.timestamp || result.memory?.created_at || '',
          updated_at: result.memory?.updated_at || result.memory?.timestamp || '',
          metadata: result.memory?.metadata || {},
          type: result.memory?.type,
          confidence: result.memory?.confidence,
          last_accessed: result.memory?.last_accessed,
        },
      })),
      count: response.count || (response.results ? response.results.length : 0),
      mode: 'ranked',
      dedup_removed: response.dedup_removed,
      query: response.query,
      sort: response.sort,
      keywords: response.keywords,
      time_window: response.time_window,
      tags: response.tags,
      exclude_tags: response.exclude_tags,
      tag_mode: response.tag_mode,
      tag_match: response.tag_match,
      state_mode: response.state_mode,
      tag_scope: response.tag_scope,
      scope_fallback: response.scope_fallback,
      recency_bias: response.recency_bias,
      score_filter: response.score_filter,
      queries: response.queries,
      vector_search: response.vector_search,
      jit_enriched_count: response.jit_enriched_count,
      query_time_ms: response.query_time_ms,
      entities: response.entities,
      expansion: response.expansion,
      entity_expansion: response.entity_expansion,
      context_priority: response.context_priority,
      state_filter: response.state_filter,
    };
  }
  async fetchMemoryById(memoryId, options) {
    // Checked here, with the API's own message, because ids that are not UUIDs
    // can resolve to other routes: "by-tag" hits GET /memory/by-tag, and ".."
    // normalizes to the viewer at "/", whose HTML would be read back as a memory.
    if (!CANONICAL_UUID_RE.test(memoryId)) {
      throw new Error(INVALID_MEMORY_ID_MESSAGE);
    }
    try {
      const response = await this._request('GET', `memory/${encodeURIComponent(memoryId)}`, undefined, options);
      return response?.memory ?? response ?? null;
    } catch (error) {
      // A missing ID is an empty result, not an error.
      if (error?.status === 404) {
        return null;
      }
      throw error;
    }
  }
  async listMemoriesByTag(tags, limit, offset, options) {
    const p = new URLSearchParams();
    tags.forEach((tag) => p.append('tags', tag));
    if (typeof limit === 'number' && limit > 0) {
      p.set('limit', String(Math.min(Math.floor(limit), 200)));
    }
    if (typeof offset === 'number' && offset > 0) {
      p.set('offset', String(Math.floor(offset)));
    }
    const response = await this._request('GET', `memory/by-tag?${p.toString()}`, undefined, options);
    const memories = Array.isArray(response.memories) ? response.memories : [];
    return {
      results: memories.map((m) => wrapMemoryAsRecallResult(m)),
      count: typeof response.count === 'number' ? response.count : memories.length,
      mode: 'enumeration',
      tags: response.tags ?? tags,
      limit: response.limit,
      offset: response.offset,
      has_more: typeof response.has_more === 'boolean' ? response.has_more : undefined,
    };
  }
  async associateMemories(args = {}, options) {
    const { associations, ...singleAssociation } = args;
    const body = Array.isArray(associations)
      ? { associations }
      : stripUndefinedValues(singleAssociation);
    const r = await this._request('POST', 'associate', body, options);
    return { success: true, message: formatAssociationResultMessage(r), response: r };
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

// Response budgeting: recall responses must stay comfortably under MCP client
// tool-response caps (~25k tokens in Claude Code). Budgeted formats
// (text/items/detailed) show a content preview, keep any stored summary as an
// additive field, collapse relations to compact stubs, and collapse metadata to
// its key list. The global budget is measured in estimated tokens; dense recall
// JSON tokenizes at ~2.5 chars/token. `format: "json"` keeps raw per-field
// passthrough, but the global budget still applies. ID fetches are never
// truncated: `memory_id` is the documented way to read a full record.
const RECALL_CONTENT_PREVIEW_CHARS = 400;
const RECALL_MAX_RELATIONS = 3;
const RECALL_RELATION_SUMMARY_CHARS = 100;
const RECALL_CHARS_PER_TOKEN = 2.5;
const DEFAULT_RECALL_TOKEN_BUDGET = 18_000;
const RESPONSE_ENVELOPE_RESERVE_TOKENS = 800;

function estimateTokens(chars) {
  return Math.ceil(chars / RECALL_CHARS_PER_TOKEN);
}

function resolveTokenBudget() {
  const raw = process.env.AUTOMEM_RECALL_TOKEN_BUDGET;
  if (raw) {
    // Strict parse: reject non-numeric suffixes ("1200foo") and fractions.
    const parsed = Number(raw.trim());
    if (Number.isInteger(parsed) && parsed > 0) {
      return parsed;
    }
  }
  return DEFAULT_RECALL_TOKEN_BUDGET;
}

function capContent(content, budgeted) {
  const text = content ?? '';
  if (!budgeted || text.length <= RECALL_CONTENT_PREVIEW_CHARS) {
    return { preview: text, truncated: false, chars: text.length };
  }
  return {
    preview: `${text.slice(0, RECALL_CONTENT_PREVIEW_CHARS)}…`,
    truncated: true,
    chars: text.length,
  };
}

function metadataKeyList(metadata) {
  if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) {
    return undefined;
  }
  const keys = Object.keys(metadata);
  return keys.length > 0 ? keys : undefined;
}

// A relation on a recall result embeds a full nested memory record. Budgeted
// formats keep only what makes the edge meaningful: the target id, edge
// type/strength, and a short summary of the target.
function relationStub(rel) {
  const memory = rel?.memory && typeof rel.memory === 'object' ? rel.memory : undefined;
  const id = memory?.id ?? rel?.id ?? rel?.memory_id;
  const rawSummary = memory?.summary ?? memory?.content ?? rel?.summary ?? rel?.content;
  const summary =
    typeof rawSummary === 'string' && rawSummary.length > 0
      ? rawSummary.length > RECALL_RELATION_SUMMARY_CHARS
        ? `${rawSummary.slice(0, RECALL_RELATION_SUMMARY_CHARS)}…`
        : rawSummary
      : undefined;
  return {
    ...(id !== undefined ? { id } : {}),
    ...(rel?.type !== undefined ? { type: rel.type } : {}),
    ...(typeof rel?.strength === 'number' ? { strength: rel.strength } : {}),
    ...(summary !== undefined ? { summary } : {}),
  };
}

function compactRelations(value) {
  if (!Array.isArray(value) || value.length === 0) {
    return {};
  }
  return {
    relations: value.slice(0, RECALL_MAX_RELATIONS).map(relationStub),
    ...(value.length > RECALL_MAX_RELATIONS ? { relations_total: value.length } : {}),
  };
}

function buildStructuredRecallItem(item, isRichFormat, budgeted, keepScoreComponents) {
  const memory = item.memory;
  const summary =
    typeof memory.summary === 'string' && memory.summary.trim().length > 0
      ? memory.summary
      : undefined;
  const { preview, truncated, chars } = capContent(memory.content, budgeted);
  // Empty content still happens on some records; fall back to summary so the
  // text channel is not a blank line. Structured `content` stays the preview
  // (possibly empty) so callers can tell the fields apart.
  const displayText = preview.trim().length > 0 ? preview : (summary ?? preview);

  const base = {
    memory_id: memory.memory_id,
    content: preview,
    ...(truncated ? { content_truncated: true, content_chars: chars } : {}),
    ...(summary !== undefined ? { summary } : {}),
    tags: memory.tags,
    importance: memory.importance,
    created_at: memory.created_at,
    updated_at: memory.updated_at,
    final_score: item.final_score,
    match_type: item.match_type,
  };
  if (!isRichFormat) {
    return { structuredItem: base, displayText, contentTruncated: truncated };
  }

  const metadataFields = budgeted
    ? (() => {
        const keys = metadataKeyList(memory.metadata);
        return keys ? { metadata_keys: keys } : {};
      })()
    : { metadata: memory.metadata };

  const structuredItem = {
    ...base,
    last_accessed: memory.last_accessed,
    ...metadataFields,
    type: memory.type,
    confidence: memory.confidence,
    ...(budgeted
      ? {}
      : {
          match_score: item.match_score,
          relation_score: item.relation_score,
          source: item.source,
        }),
    ...(keepScoreComponents ? { score_components: item.score_components } : {}),
    ...(budgeted ? compactRelations(item.relations) : { relations: item.relations }),
    deduped_from: item.deduped_from,
    expanded_from_entity: item.expanded_from_entity,
    outside_tag_scope: item.outside_tag_scope,
    jit_enriched: item.jit_enriched,
    state_replaces: item.state_replaces,
  };
  return { structuredItem, displayText, contentTruncated: truncated };
}

function buildStructuredEnvelope(recallResult) {
  const results = recallResult.results || [];
  return {
    count: recallResult.count ?? results.length,
    ...(recallResult.mode ? { mode: recallResult.mode } : {}),
    ...(typeof recallResult.has_more === 'boolean' ? { has_more: recallResult.has_more } : {}),
    ...(typeof recallResult.limit === 'number' ? { limit: recallResult.limit } : {}),
    ...(typeof recallResult.offset === 'number' ? { offset: recallResult.offset } : {}),
    ...(typeof recallResult.dedup_removed === 'number'
      ? { dedup_removed: recallResult.dedup_removed }
      : {}),
    ...(recallResult.query ? { query: recallResult.query } : {}),
    ...(recallResult.sort ? { sort: recallResult.sort } : {}),
    ...(recallResult.keywords ? { keywords: recallResult.keywords } : {}),
    ...(recallResult.time_window ? { time_window: recallResult.time_window } : {}),
    ...(recallResult.tags ? { tags: recallResult.tags } : {}),
    ...(recallResult.exclude_tags ? { exclude_tags: recallResult.exclude_tags } : {}),
    ...(recallResult.tag_mode ? { tag_mode: recallResult.tag_mode } : {}),
    ...(recallResult.tag_match ? { tag_match: recallResult.tag_match } : {}),
    ...(recallResult.state_mode ? { state_mode: recallResult.state_mode } : {}),
    ...(recallResult.tag_scope ? { tag_scope: recallResult.tag_scope } : {}),
    ...(typeof recallResult.scope_fallback === 'boolean'
      ? { scope_fallback: recallResult.scope_fallback }
      : {}),
    ...(recallResult.recency_bias ? { recency_bias: recallResult.recency_bias } : {}),
    ...(recallResult.score_filter ? { score_filter: recallResult.score_filter } : {}),
    ...(recallResult.queries ? { queries: recallResult.queries } : {}),
    ...(recallResult.vector_search ? { vector_search: recallResult.vector_search } : {}),
    ...(typeof recallResult.jit_enriched_count === 'number'
      ? { jit_enriched_count: recallResult.jit_enriched_count }
      : {}),
    ...(typeof recallResult.query_time_ms === 'number'
      ? { query_time_ms: recallResult.query_time_ms }
      : {}),
    ...(recallResult.entities ? { entities: recallResult.entities } : {}),
    ...(recallResult.expansion ? { expansion: recallResult.expansion } : {}),
    ...(recallResult.entity_expansion ? { entity_expansion: recallResult.entity_expansion } : {}),
    ...(recallResult.context_priority ? { context_priority: recallResult.context_priority } : {}),
    ...(recallResult.state_filter ? { state_filter: recallResult.state_filter } : {}),
  };
}

function renderTextBlock(item, preview, index) {
  const memory = item.memory;
  const tags = memory.tags?.length ? ` [${memory.tags.join(', ')}]` : '';
  const importance =
    typeof memory.importance === 'number' ? ` (importance: ${memory.importance})` : '';
  const score = typeof item.final_score === 'number' ? ` score=${item.final_score.toFixed(3)}` : '';
  const matchType = item.match_type ? ` [${item.match_type}]` : '';
  const relationNote =
    Array.isArray(item.relations) && item.relations.length
      ? ` relations=${item.relations.length}`
      : '';
  const dedupNote =
    Array.isArray(item.deduped_from) && item.deduped_from.length
      ? ` (deduped x${item.deduped_from.length})`
      : '';
  const entityNote = item.expanded_from_entity ? ` [via entity: ${item.expanded_from_entity}]` : '';
  // The stored date gets its own line: without it a caller replaying recall
  // text cannot tell a note written today from one written weeks ago.
  const updatedNote = memory.updated_at ? `  Updated: ${memory.updated_at}` : '';
  return `${index + 1}. ${preview}${tags}${importance}${score}${matchType}${relationNote}${entityNote}${dedupNote}\n   ID: ${
    memory.memory_id
  }\n   Created: ${memory.created_at}${updatedNote}`;
}

function renderDetailedBlock(item, preview) {
  const memory = item.memory;
  const lines = [preview, `  ID: ${memory.memory_id}`];
  if (memory.type) lines.push(`  Type: ${memory.type}`);
  lines.push(`  Created: ${memory.created_at}`);
  if (memory.updated_at) lines.push(`  Updated: ${memory.updated_at}`);
  if (memory.last_accessed) lines.push(`  Accessed: ${memory.last_accessed}`);
  if (typeof memory.importance === 'number') {
    lines.push(`  Importance: ${memory.importance.toFixed(3)}`);
  }
  if (typeof memory.confidence === 'number') {
    lines.push(`  Confidence: ${memory.confidence.toFixed(3)}`);
  }
  if (memory.tags?.length) lines.push(`  Tags: ${memory.tags.join(', ')}`);
  if (typeof item.final_score === 'number') {
    lines.push(`  Score: ${item.final_score.toFixed(3)}`);
  }
  if (item.match_type) lines.push(`  Match: ${item.match_type}`);
  return lines.join('\n');
}

async function buildRecallMemoryResponse(client, recallArgs, requestOptions) {
  const recallResult = await client.recallMemory(recallArgs, requestOptions);
  const results = recallResult.results || [];
  const format = recallArgs.format || 'text';
  const isRichFormat = format === 'detailed' || format === 'json';
  const isIdFetch = recallResult.mode === 'id_fetch' || Boolean(recallArgs.memory_id);
  // json keeps raw per-field passthrough; id fetches are never truncated.
  const budgeted = !isIdFetch && format !== 'json';
  const keepScoreComponents = format === 'json' || isIdFetch;

  if (results.length === 0) {
    return {
      content: [
        {
          type: 'text',
          text: 'No memories found matching your query.',
        },
      ],
      structuredContent: {
        results: [],
        ...buildStructuredEnvelope(recallResult),
      },
    };
  }

  const perItem = results.map((item, index) => {
    const { structuredItem, displayText, contentTruncated } = buildStructuredRecallItem(
      item,
      isRichFormat,
      budgeted,
      keepScoreComponents
    );
    let textBlock = '';
    if (format === 'items') {
      textBlock = `[${item.memory.memory_id}] ${displayText}`;
    } else if (format === 'detailed') {
      textBlock = renderDetailedBlock(item, displayText);
    } else if (format !== 'json') {
      textBlock = renderTextBlock(item, displayText, index);
    }
    const structuredLength = JSON.stringify(structuredItem)?.length ?? 0;
    // json repeats the structured payload in the text channel pretty-printed;
    // measure that length directly (nesting can inflate it well past 2x).
    const cost =
      format === 'json'
        ? structuredLength + (JSON.stringify(structuredItem, null, 2)?.length ?? 0)
        : structuredLength + textBlock.length;
    return { structuredItem, textBlock, cost, contentTruncated };
  });

  // Global budget: always keep the first result; keep the rest while in budget.
  const tokenBudget = resolveTokenBudget();
  const kept = [];
  let runningTokens = RESPONSE_ENVELOPE_RESERVE_TOKENS;
  if (isIdFetch) {
    kept.push(...perItem);
  } else {
    for (const entry of perItem) {
      const entryTokens = estimateTokens(entry.cost);
      if (kept.length > 0 && runningTokens + entryTokens > tokenBudget) {
        break;
      }
      kept.push(entry);
      runningTokens += entryTokens;
    }
  }
  const omitted = perItem.length - kept.length;

  const structuredContent = {
    results: kept.map((entry) => entry.structuredItem),
    ...buildStructuredEnvelope(recallResult),
    ...(omitted > 0
      ? {
          truncation: {
            applied: true,
            omitted_results: omitted,
            reason: 'response_token_budget',
          },
        }
      : {}),
  };

  const notes = [];
  if ((recallResult.dedup_removed || 0) > 0) {
    notes.push(`${recallResult.dedup_removed} duplicates removed`);
  }
  if (recallResult.entity_expansion?.enabled && recallResult.entity_expansion.expanded_count > 0) {
    notes.push(
      `${recallResult.entity_expansion.expanded_count} via entity expansion (${
        recallResult.entity_expansion.entities_found?.join(', ') || 'entities found'
      })`
    );
  }
  if (recallResult.expansion?.enabled && recallResult.expansion.expanded_count > 0) {
    notes.push(`${recallResult.expansion.expanded_count} via relation expansion`);
  }
  if (recallResult.state_filter) {
    notes.push(
      `state filter suppressed ${recallResult.state_filter.suppressed_count}, replacements ${recallResult.state_filter.replacement_count}`
    );
  }
  if (recallResult.scope_fallback) {
    notes.push('scope fallback included outside-scope results');
  }
  const filteredCount = recallResult.score_filter?.filtered_count;
  if (typeof filteredCount === 'number' && filteredCount > 0) {
    notes.push(`score filter removed ${filteredCount}`);
  }
  if (recallResult.mode === 'enumeration') {
    const offset = recallResult.offset ?? 0;
    const limit = recallResult.limit ?? results.length;
    const pageSuffix = recallResult.has_more ? ' — more pages available' : '';
    notes.push(`enumeration page: offset ${offset}, limit ${limit}${pageSuffix}`);
  }
  const notesSuffix = notes.length > 0 ? ` (${notes.join('; ')})` : '';

  const anyContentTruncated = kept.some((entry) => entry.contentTruncated);
  const trailerParts = [];
  if (omitted > 0) {
    trailerParts.push(
      `Response budget: showing ${kept.length} of ${perItem.length} results; ${omitted} omitted.`
    );
  }
  if (anyContentTruncated) {
    trailerParts.push(
      'Content shown as previews — fetch full records with recall_memory({ memory_id: "<id>" }).'
    );
  }
  const trailer = trailerParts.length > 0 ? `\n\n[${trailerParts.join(' ')}]` : '';

  if (format === 'json') {
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(structuredContent, null, 2),
        },
      ],
      structuredContent,
    };
  }

  if (format === 'items') {
    const itemBlocks = kept.map((entry) => ({
      type: 'text',
      text: entry.textBlock,
    }));
    if (trailer) {
      itemBlocks.push({ type: 'text', text: trailer.trim() });
    }
    return {
      content: itemBlocks,
      structuredContent,
    };
  }

  const joinedBlocks = kept.map((entry) => entry.textBlock).join('\n\n');
  const showingSuffix = omitted > 0 ? ` (showing ${kept.length})` : '';
  return {
    content: [
      {
        type: 'text',
        text: `Found ${results.length} memories${showingSuffix}${notesSuffix}:\n\n${joinedBlocks}${trailer}`,
      },
    ],
    structuredContent,
  };
}

// Copied from mcp-automem's src/mcp-surface.ts. test/recall-parity.test.js
// deep-compares it with the installed package, so edit it there first.
const RECALL_MEMORY_TOOL = {
  name: 'recall_memory',
  title: 'Recall Memory',
  description: `Recall memories from AutoMem in one of three modes. The mode is selected by which params you pass.

**Mode 1 — ID fetch:** pass \`memory_id\` to retrieve a single memory by ID. All other params are ignored. Routes to GET /memory/{id} and updates last_accessed.

**Mode 2 — Tag enumeration:** pass \`tags\` + \`exhaustive: true\` for paginated exact-match listing (NOT ranked retrieval). Use this for cleanup/audit workflows where ranked retrieval silently undercounts large tag sets. Pair with \`limit\` (≤200) and \`offset\`. Returns \`has_more\`/\`limit\`/\`offset\` page metadata. Tag matching is exact, case-insensitive, any-of mode — \`tag_match: "prefix"\` and \`tag_mode: "all"\` are rejected in this mode.

**Mode 3 — Ranked retrieval (default):** hybrid search across vector, keyword, tags, recency, and optional graph expansion. The primary tool for finding relevant context. By default, ranked recall requests current active memories only; set \`current_only: false\` for audits.

**When to use ranked (mode 3):**
- At conversation start: recall context about the current project/topic
- Before making decisions: check for past decisions on similar topics
- When debugging: search for similar past errors and their solutions
- For complex questions: use \`expand_entities\` for multi-hop reasoning

**When to use enumeration (mode 2):** when you need to know *how many* memories carry a tag, or to walk all of them for cleanup/migration. Ranked recall ignores low-importance hits — enumeration does not.

**Examples:**
- recall_memory({ query: "database architecture decisions", tags: ["my-project"], limit: 5 })
- recall_memory({ memory_id: "abc123" })  // Mode 1
- recall_memory({ tags: ["benchmark-test"], exhaustive: true, limit: 50 })  // Mode 2 (add offset for later pages)
- recall_memory({ query: "auth", exclude_tags: ["deprecated"] })  // Mode 3 with exclusion
- recall_memory({ query: "What is Sarah's sister's job?", expand_entities: true })  // Mode 3 multi-hop`,
  annotations: {
    title: 'Recall Memory',
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: false,
  },
  _meta: { 'anthropic/alwaysLoad': true },
  inputSchema: {
    type: 'object',
    properties: {
      memory_id: {
        type: 'string',
        description:
          'MODE: ID fetch. When set, fetches the single memory by ID and IGNORES all other params. Routes to GET /memory/{id}; updates last_accessed.',
      },
      exhaustive: {
        type: 'boolean',
        description:
          'MODE: tag enumeration. When true, requires non-empty `tags`. Routes to GET /memory/by-tag for paginated exact-match listing — NOT ranked retrieval. Use for cleanup/audit workflows where ranked recall undercounts. `limit` is clamped to 200. `tag_match: "prefix"` and `tag_mode: "all"` are rejected in this mode.',
      },
      exclude_tags: {
        type: 'array',
        items: { type: 'string' },
        description:
          'Ranked-mode only. Tags to exclude from results (any match excludes). Independent of `tag_match` — supports both exact and prefix matching internally on the server.',
      },
      query: {
        type: 'string',
        description: "Semantic search query (natural language). Describe what you're looking for.",
      },
      queries: {
        type: 'array',
        items: { type: 'string' },
        description: 'Multiple queries for broader recall. Results are deduplicated server-side.',
      },
      embedding: {
        type: 'array',
        items: { type: 'number' },
        description: 'Optional embedding vector for direct similarity search',
      },
      limit: {
        type: 'integer',
        minimum: 1,
        maximum: 200,
        default: 5,
        description:
          'Max memories to return. Schema allows 1–200; in enumeration mode (`exhaustive: true`) the server honors up to 200, while ranked mode is typically clamped server-side to ~50. Default 5.',
      },
      time_query: {
        type: 'string',
        description: 'Natural language time filter: "today", "yesterday", "last week", "last 30 days"',
      },
      start: {
        type: 'string',
        description: 'ISO timestamp lower bound (alternative to time_query)',
      },
      end: {
        type: 'string',
        description: 'ISO timestamp upper bound',
      },
      tags: {
        type: 'array',
        items: { type: 'string' },
        description: 'Filter by tags. Use project name as first tag for scoping.',
      },
      tag_mode: {
        type: 'string',
        enum: ['any', 'all'],
        description: '"any" matches memories with any tag (default), "all" requires all tags',
      },
      tag_match: {
        type: 'string',
        enum: ['exact', 'prefix'],
        description: '"exact" for exact tag match (default), "prefix" for starts-with matching',
      },
      expand_entities: {
        type: 'boolean',
        description:
          'Enable multi-hop reasoning via entity expansion. Finds memories about people/places mentioned in seed results. Use for "What is X\'s sister\'s job?" type questions.',
      },
      expand_relations: {
        type: 'boolean',
        description: 'Follow graph relationships from seed results to find related memories.',
      },
      expand_respect_tags: {
        type: 'boolean',
        description:
          'Ranked-mode only. When true, graph/entity expansion stays within the original tag scope; when false, expansion may include related context outside the tags.',
      },
      auto_decompose: {
        type: 'boolean',
        description: 'Auto-extract entities and topics from query to generate supplementary searches.',
      },
      expansion_limit: {
        type: 'integer',
        minimum: 1,
        maximum: 500,
        default: 25,
        description: 'Max total expanded memories (default: 25)',
      },
      relation_limit: {
        type: 'integer',
        minimum: 1,
        maximum: 200,
        default: 5,
        description: 'Max relations to follow per seed memory (default: 5)',
      },
      expand_min_importance: {
        type: 'number',
        minimum: 0,
        maximum: 1,
        description:
          'Minimum importance score for expanded results. Filters out low-relevance memories during graph/entity expansion. Recommended: 0.3-0.5 for broad context, 0.6-0.8 for focused results. Seed results are never filtered, only expanded ones.',
      },
      expand_min_strength: {
        type: 'number',
        minimum: 0,
        maximum: 1,
        description:
          'Minimum relation strength to follow during graph expansion. Only traverses edges above this threshold. Recommended: 0.3 for exploratory, 0.6+ for high-confidence connections only. Does not affect entity expansion.',
      },
      current_only: {
        type: 'boolean',
        default: true,
        description:
          'Ranked-mode only. When true, server suppresses archived, not-yet-valid, expired, invalidated, or superseded memories from active context.',
      },
      state_debug: {
        type: 'boolean',
        default: false,
        description:
          'Ranked-mode only. Include state-filter suppression/replacement IDs and reasons when current_only is true.',
      },
      state_mode: {
        type: 'string',
        enum: ['current', 'history'],
        description:
          'Ranked-mode only. `current` returns active memories; `history` allows superseded/invalidated memories for audit timelines. Prefer this over current_only for new clients.',
      },
      recency_bias: {
        type: 'string',
        enum: ['auto', 'on', 'off'],
        description:
          'Ranked-mode only. Controls service recency boosting: auto lets the service infer, on forces boosting, off disables it.',
      },
      scope_fallback: {
        type: 'boolean',
        description:
          'Ranked-mode only. Allow fallback outside the requested tag scope when scoped recall has weak evidence; diagnostics report tag_scope and outside_tag_scope.',
      },
      min_score: {
        type: 'number',
        minimum: 0,
        maximum: 1,
        description: 'Ranked-mode only. Minimum final score threshold before results are returned.',
      },
      adaptive_floor: {
        type: 'boolean',
        description: "Ranked-mode only. Enable the service's adaptive score floor when filtering weak matches.",
      },
      context: {
        type: 'string',
        description: 'Context label (e.g., "coding-style", "architecture"). Boosts matching preferences.',
      },
      language: {
        type: 'string',
        description:
          'Programming language hint (e.g., "python", "typescript"). Prioritizes language-specific memories.',
      },
      active_path: {
        type: 'string',
        description: 'Current file path for language auto-detection (e.g., "src/auth.ts")',
      },
      context_tags: {
        type: 'array',
        items: { type: 'string' },
        description: 'Priority tags to boost in results (e.g., ["coding-style", "preferences"])',
      },
      context_types: {
        type: 'array',
        items: { type: 'string' },
        description: 'Priority memory types to boost (e.g., ["Style", "Preference"])',
      },
      priority_ids: {
        type: 'array',
        items: { type: 'string' },
        description: 'Specific memory IDs to ensure are included in results',
      },
      per_query_limit: {
        type: 'integer',
        minimum: 1,
        maximum: 50,
        description: 'Per-query result limit when using queries[] (default: 5)',
      },
      sort: {
        type: 'string',
        enum: ['score', 'time_desc', 'time_asc', 'updated_desc', 'updated_asc'],
        description: 'Result ordering (use time_* for chronological recaps)',
      },
      format: {
        type: 'string',
        enum: ['text', 'items', 'detailed', 'json'],
        default: 'text',
        description:
          'Output format: text (default), items (one block per memory), detailed (adds type/confidence/metadata keys/relation stubs), json (raw per-memory fields incl. full content/metadata/relations; whole-response token budget still applies). text/items/detailed show a content preview (default 400 chars) and keep any stored summary as an additive field — fetch a full record via memory_id.',
      },
      offset: {
        type: 'integer',
        minimum: 0,
        description: 'Result offset for pagination',
      },
    },
  },
  outputSchema: {
    type: 'object',
    properties: {
      count: {
        type: 'integer',
        description: 'Number of memories returned',
      },
      mode: {
        type: 'string',
        enum: ['ranked', 'enumeration', 'id_fetch'],
        description: 'Mode that produced the result.',
      },
      has_more: {
        type: 'boolean',
        description: 'Enumeration mode only: true if more pages exist past `offset + limit`.',
      },
      limit: {
        type: 'integer',
        description: 'Enumeration mode only: page size used for this response.',
      },
      offset: {
        type: 'integer',
        description: 'Enumeration mode only: offset used for this response.',
      },
      results: {
        type: 'array',
        description: 'Array of matching memories with scores',
        items: {
          type: 'object',
          properties: {
            memory_id: { type: 'string' },
            summary: {
              type: 'string',
              description:
                'Stored 1-2 sentence summary when the server provides one. Additive in budgeted formats; does not replace content.',
            },
            content: {
              type: 'string',
              description: 'Memory content (preview-capped in budgeted formats).',
            },
            content_truncated: {
              type: 'boolean',
              description:
                'True when content is a preview; fetch the full record via recall_memory({ memory_id }).',
            },
            content_chars: {
              type: 'integer',
              description: 'Original content length when content was previewed.',
            },
            tags: { type: 'array', items: { type: 'string' } },
            importance: { type: 'number' },
            final_score: { type: 'number' },
            match_type: { type: 'string' },
            created_at: { type: 'string' },
            updated_at: { type: 'string' },
            deduped_from: {
              type: 'array',
              items: { type: 'string' },
              description: 'Result IDs merged into this result during multi-query deduplication.',
            },
            outside_tag_scope: {
              type: 'boolean',
              description: 'True when scope_fallback admitted this result outside the requested tag scope.',
            },
            jit_enriched: {
              type: 'boolean',
              description: 'True when the service enriched the memory during recall.',
            },
            state_replaces: {
              type: 'string',
              description: 'ID of the suppressed memory this result replaced during current-state filtering.',
            },
          },
        },
      },
      truncation: {
        type: 'object',
        description:
          'Present when trailing results were dropped to fit the response budget: { applied, omitted_results, reason }.',
      },
      dedup_removed: {
        type: 'integer',
        description: 'Number of duplicate results removed (when using multiple queries)',
      },
      query: {
        type: 'string',
        description: 'Query text executed by ranked recall.',
      },
      sort: {
        type: 'string',
        description: 'Sort mode applied by the service.',
      },
      exclude_tags: {
        type: 'array',
        items: { type: 'string' },
        description: 'Tags excluded from ranked recall.',
      },
      state_filter: {
        type: 'object',
        description:
          'Current-state filtering diagnostics. Includes aggregate counts by default and detailed IDs/reasons only when state_debug=true.',
      },
      state_mode: {
        type: 'string',
        enum: ['current', 'history'],
        description: 'State mode applied by ranked recall.',
      },
      tag_scope: {
        type: 'object',
        description: 'Tag-scope diagnostics including whether scoped evidence was strong enough.',
      },
      scope_fallback: {
        type: 'boolean',
        description: 'True when recall allowed outside-scope fallback results.',
      },
      recency_bias: {
        type: 'string',
        enum: ['auto', 'on', 'off'],
        description: 'Recency bias mode applied by the service.',
      },
      score_filter: {
        type: 'object',
        description: 'Score filtering diagnostics such as min_score, adaptive_floor, and filtered_count.',
      },
      queries: {
        type: 'array',
        items: { type: 'string' },
        description: 'Query variants executed by the service.',
      },
      vector_search: {
        type: 'object',
        description: 'Vector-search diagnostics from the service.',
      },
      jit_enriched_count: {
        type: 'integer',
        description: 'Number of memories enriched inline during recall.',
      },
      query_time_ms: {
        type: 'number',
        description: 'Service recall latency in milliseconds.',
      },
      entities: {
        type: 'array',
        items: { type: 'object' },
        description: 'Entity identity diagnostics injected by the service.',
      },
    },
    required: ['count', 'results'],
  },
};

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
    RECALL_MEMORY_TOOL,
    {
      name: 'associate_memories',
      description: 'Create one association or a batch of associations between memories',
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
          associations: {
            type: 'array',
            minItems: 1,
            maxItems: 500,
            description: 'Batch of associations to create. Each item uses memory1_id, memory2_id, type, and strength.',
            items: {
              type: 'object',
              properties: {
                memory1_id: { type: 'string', description: 'ID of the source memory' },
                memory2_id: { type: 'string', description: 'ID of the target memory' },
                type: {
                  type: 'string',
                  enum: RELATION_TYPES,
                  description: 'Relationship type between the two memories',
                },
                strength: {
                  type: 'number',
                  minimum: 0,
                  maximum: 1,
                  description: 'Relationship strength (0-1)',
                },
              },
              required: ['memory1_id', 'memory2_id', 'type', 'strength'],
              additionalProperties: true,
            },
          },
        },
        anyOf: [
          { required: ['memory1_id', 'memory2_id', 'type', 'strength'] },
          { required: ['associations'] },
        ],
        additionalProperties: true,
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
          return await buildRecallMemoryResponse(client, args || {}, { requestId });
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

  // Liveness probe: returns 200 whenever the Node process is able to serve HTTP.
  // Upstream AutoMem status is reported in the body for observability but does
  // NOT gate this endpoint — a bridge should be deployable and able to return
  // errors to clients even when its upstream is temporarily unavailable.
  // For strict upstream-gated readiness, point your orchestrator at /ready.
  app.get('/health', async (req, res) => {
    if (!healthState.checked_at) {
      void probeUpstreamHealth('request');
    }

    res.status(200).json({
      status: healthState.status,
      transports: ['streamable-http', 'sse'],
      endpoints: { streamableHttp: '/mcp', sse: '/mcp/sse' },
      upstream: healthState.upstream,
      upstream_error: healthState.error,
      upstream_details: healthState.details,
      checked_at: healthState.checked_at,
      timestamp: new Date().toISOString(),
      request_id: req.requestId,
    });
  });

  // Readiness probe: 200 iff upstream is healthy, 503 otherwise.
  // Use this when the orchestrator should block traffic/deploy on upstream
  // availability. Not recommended as Railway's healthcheckPath for this bridge.
  app.get('/ready', async (req, res) => {
    await probeUpstreamHealth('request');

    res.status(healthState.status === 'healthy' ? 200 : 503).json({
      status: healthState.status,
      upstream: healthState.upstream,
      upstream_error: healthState.error,
      checked_at: healthState.checked_at,
      timestamp: new Date().toISOString(),
      request_id: req.requestId,
    });
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
