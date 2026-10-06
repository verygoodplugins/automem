/**
 * recall_memory parity guard. Runs in CI's node-test job, so no live stack.
 *
 * The remote bridge's recall_memory is a port of the stdio package's
 * (@verygoodplugins/mcp-automem). This drives the bridge over streamable HTTP
 * and the package's real MCP server over an in-memory transport, both against
 * one canned AutoMem API, and requires the same tool definition, the same
 * upstream requests, and the same client-visible result for every case.
 *
 * Both sides run through an SDK Client that has listed tools first, so
 * structuredContent is also validated against each side's outputSchema.
 *
 * When this fails after bumping the devDependency, the package changed
 * recall_memory: port the change into server.js rather than loosening this.
 */
import test, { after, before } from 'node:test';
import assert from 'node:assert/strict';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { createAutoMemMcpServer } from '@verygoodplugins/mcp-automem/dist/mcp-surface.js';
import { AutoMemClient as StdioAutoMemClient } from '@verygoodplugins/mcp-automem/dist/automem-client.js';
import { createApp } from '../server.js';
import {
  FAKE_API_URL,
  MEMORIES,
  UUID_A,
  UUID_B,
  UUID_C,
  UUID_MISSING,
  createFakeAutoMem,
  richRecallResponse,
} from '../parity/fake-automem.js';

const API_TOKEN = 'parity-token';
const fake = createFakeAutoMem();

let remote;
let stdio;
const cleanups = [];

before(async () => {
  const originalFetch = globalThis.fetch;
  const originalConsoleError = console.error;
  const savedEnv = {
    AUTOMEM_API_URL: process.env.AUTOMEM_API_URL,
    AUTOMEM_API_TOKEN: process.env.AUTOMEM_API_TOKEN,
  };

  globalThis.fetch = (url, init) =>
    String(url).startsWith(FAKE_API_URL) ? fake.handle(String(url), init) : originalFetch(url, init);
  // The stdio client logs every upstream error to stderr; expected errors here
  // would bury real failures.
  console.error = () => {};
  process.env.AUTOMEM_API_URL = FAKE_API_URL;
  process.env.AUTOMEM_API_TOKEN = API_TOKEN;
  cleanups.push(() => {
    globalThis.fetch = originalFetch;
    console.error = originalConsoleError;
    for (const [key, value] of Object.entries(savedEnv)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  });

  // Remote: the bridge, booted in-process and driven over streamable HTTP.
  const httpServer = await new Promise((resolve) => {
    const s = createApp().listen(0, '127.0.0.1', () => resolve(s));
  });
  cleanups.push(() => new Promise((resolve) => httpServer.close(resolve)));
  remote = new Client({ name: 'recall-parity', version: '1.0.0' }, {});
  const remoteTransport = new StreamableHTTPClientTransport(
    new URL(`http://127.0.0.1:${httpServer.address().port}/mcp`),
    { requestInit: { headers: { Authorization: `Bearer ${API_TOKEN}` } } }
  );
  cleanups.push(() => remote.close());
  await remote.connect(remoteTransport);

  // stdio: the published package's own MCP server and client. No retries, so
  // an expected API error fails fast instead of backing off.
  const stdioServer = createAutoMemMcpServer({
    client: new StdioAutoMemClient({ endpoint: FAKE_API_URL, apiKey: API_TOKEN, maxRetries: 0 }),
    name: 'mcp-automem',
    version: 'parity',
  });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await stdioServer.connect(serverTransport);
  stdio = new Client({ name: 'recall-parity', version: '1.0.0' }, {});
  cleanups.push(() => stdio.close());
  await stdio.connect(clientTransport);

  // Listing tools caches each side's outputSchema in its Client, which turns
  // on structuredContent validation for every callTool below.
  await remote.listTools();
  await stdio.listTools();
  fake.takeRequests();
});

after(async () => {
  for (const cleanup of cleanups.reverse()) {
    await Promise.resolve(cleanup()).catch(() => {});
  }
});

// The message an error carries, without its transport framing: the remote's
// "AutoMem error: " prefix and request id suffix (allowlisted in
// docs/MCP_TRANSPORT_PARITY.md), or the SDK's "MCP error <code>: " prefix.
function errorCore(result) {
  if (result.thrown) return result.thrown.replace(/^MCP error -?\d+: /, '');
  const text = (result.content || []).map((c) => c.text ?? '').join('\n');
  return text
    .replace(/\s*\(request_id: [^)]*\)\s*$/, '')
    .replace(/^(AutoMem error|Error): /, '');
}

// The in-memory transport hands objects over by reference, so undefined fields
// survive that a real stdio pipe would drop when it serializes to JSON.
const onTheWire = (result) => JSON.parse(JSON.stringify(result));

async function callRecall(client, args) {
  try {
    return onTheWire(await client.callTool({ name: 'recall_memory', arguments: args }));
  } catch (error) {
    // stdio's handler returns the recall promise without awaiting it
    // (mcp-automem src/mcp-surface.ts), so its catch never runs and a recall
    // error reaches the client as a JSON-RPC error. The remote awaits and
    // returns an isError result, which is what MCP specifies for tool errors.
    return { thrown: error.message };
  }
}

/**
 * Call recall_memory on both transports and return what each client saw plus
 * the API requests each one made.
 */
async function callBoth(args, { env = {}, recall } = {}) {
  const saved = Object.fromEntries(Object.keys(env).map((k) => [k, process.env[k]]));
  Object.assign(process.env, env);
  fake.setRecallResponder(recall);
  try {
    fake.takeRequests();
    const remoteResult = await callRecall(remote, args);
    const remoteRequests = fake.takeRequests();
    const stdioResult = await callRecall(stdio, args);
    const stdioRequests = fake.takeRequests();
    return { remoteResult, remoteRequests, stdioResult, stdioRequests };
  } finally {
    fake.setRecallResponder();
    for (const [key, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
}

/** Same requests, same result. Errors compare on the message, without the transport framing. */
async function assertParity(args, options) {
  const run = await callBoth(args, options);
  assert.equal(run.remoteResult.thrown, undefined, 'remote must report tool errors as isError results');
  assert.deepStrictEqual(run.remoteRequests, run.stdioRequests, 'upstream requests differ');
  if (run.stdioResult.isError || run.stdioResult.thrown) {
    assert.equal(run.remoteResult.isError, true, 'stdio errored but remote did not');
    assert.equal(errorCore(run.remoteResult), errorCore(run.stdioResult));
  } else {
    assert.deepStrictEqual(run.remoteResult, run.stdioResult);
  }
  return run;
}

test('recall_memory tool definition is identical across transports', async () => {
  const [remoteTools, stdioTools] = await Promise.all([remote.listTools(), stdio.listTools()]);
  const pick = (list) => list.tools.find((t) => t.name === 'recall_memory');
  assert.deepStrictEqual(pick(remoteTools), pick(stdioTools));
});

// --- Mode 1: ID fetch ---------------------------------------------------------

test('ID fetch returns that one memory, in every format', async () => {
  for (const format of [undefined, 'text', 'items', 'detailed', 'json']) {
    const { remoteResult, remoteRequests } = await assertParity({
      memory_id: UUID_A,
      ...(format ? { format } : {}),
    });
    assert.deepStrictEqual(remoteRequests, [`GET /memory/${UUID_A}`]);
    assert.equal(remoteResult.structuredContent.mode, 'id_fetch');
    assert.deepStrictEqual(
      remoteResult.structuredContent.results.map((r) => r.memory_id),
      [UUID_A]
    );
  }
});

test('ID fetch ignores every other param and never falls through to /recall', async () => {
  const { remoteRequests } = await assertParity({
    memory_id: `  ${UUID_A}  `,
    query: 'something else entirely',
    tags: ['unrelated'],
    exclude_tags: ['db'],
    exhaustive: true,
    limit: 1,
    current_only: false,
  });
  assert.deepStrictEqual(remoteRequests, [`GET /memory/${UUID_A}`]);
});

test('ID fetch never truncates content', async () => {
  const { remoteResult } = await assertParity({ memory_id: UUID_B });
  const [item] = remoteResult.structuredContent.results;
  assert.equal(item.content, MEMORIES[1].content);
  assert.ok(item.content.length > 400);
  assert.equal(item.content_truncated, undefined);
});

test('ID fetch of an unknown id is an empty result, not an error', async () => {
  const { remoteResult } = await assertParity({ memory_id: UUID_MISSING });
  assert.equal(remoteResult.isError, undefined);
  assert.equal(remoteResult.structuredContent.count, 0);
  assert.equal(remoteResult.structuredContent.mode, 'id_fetch');
});

test('ID fetch of a non-UUID is an error with the API message', async () => {
  // The remote rejects before calling the API (an id like ".." would otherwise
  // resolve to the viewer route); stdio lets the API reject it. Same message.
  const run = await callBoth({ memory_id: '67c0f41f' });
  assert.deepStrictEqual(run.remoteRequests, []);
  assert.deepStrictEqual(run.stdioRequests, ['GET /memory/67c0f41f']);
  assert.equal(run.remoteResult.isError, true);
  assert.ok(run.stdioResult.isError || run.stdioResult.thrown, 'stdio must fail too');
  assert.equal(errorCore(run.remoteResult), 'memory_id must be a valid UUID');
  assert.equal(errorCore(run.remoteResult), errorCore(run.stdioResult));
});

test('ID fetch forwards every UUID spelling the API accepts, as stdio does', async () => {
  // Python's uuid.UUID() takes braces, a urn:uuid: prefix, missing hyphens and
  // int(s, 16) grammar such as 0x and underscores. The remote's pre-check must
  // not be stricter than the API it guards.
  const intGrammar = [`0x${'a'.repeat(30)}`, `${'a'.repeat(16)}_${'a'.repeat(15)}`];
  for (const memoryId of [`{${UUID_A}}`, `urn:uuid:${UUID_A}`, UUID_A.replaceAll('-', ''), ...intGrammar]) {
    const { remoteRequests } = await assertParity({ memory_id: memoryId });
    assert.deepStrictEqual(remoteRequests, [`GET /memory/${encodeURIComponent(memoryId)}`]);
  }
});

test('a blank memory_id means no ID fetch, on both transports', async () => {
  await assertParity({ memory_id: '   ', query: 'parity' });
});

// --- Mode 2: tag enumeration ---------------------------------------------------

test('enumeration pages through GET /memory/by-tag with has_more', async () => {
  const first = await assertParity({ tags: ['parity'], exhaustive: true, limit: 2 });
  assert.deepStrictEqual(first.remoteRequests, ['GET /memory/by-tag?limit=2&tags=parity']);
  assert.equal(first.remoteResult.structuredContent.mode, 'enumeration');
  assert.equal(first.remoteResult.structuredContent.has_more, true);

  const second = await assertParity({ tags: ['parity'], exhaustive: true, limit: 2, offset: 2 });
  assert.deepStrictEqual(second.remoteRequests, [
    'GET /memory/by-tag?limit=2&offset=2&tags=parity',
  ]);
  assert.equal(second.remoteResult.structuredContent.has_more, false);
  assert.deepStrictEqual(
    second.remoteResult.structuredContent.results.map((r) => r.memory_id),
    [UUID_C]
  );
});

test('enumeration clamps limit to 200, trims tags, and keeps the API default limit', async () => {
  const clamped = await assertParity({ tags: ['parity'], exhaustive: true, limit: 500 });
  assert.deepStrictEqual(clamped.remoteRequests, ['GET /memory/by-tag?limit=200&tags=parity']);

  const defaulted = await assertParity({ tags: [' PARITY ', ''], exhaustive: true });
  assert.deepStrictEqual(defaulted.remoteRequests, ['GET /memory/by-tag?tags=PARITY']);
});

test('enumeration renders identically in every format', async () => {
  for (const format of ['text', 'items', 'detailed', 'json']) {
    await assertParity({
      tags: ['parity'],
      exhaustive: true,
      tag_mode: 'any',
      tag_match: 'exact',
      exclude_tags: [],
      format,
    });
  }
});

test('enumeration rejects what GET /memory/by-tag cannot honor, without calling the API', async () => {
  const rejected = [
    { exhaustive: true },
    { exhaustive: true, tags: ['  '] },
    { exhaustive: true, tags: ['parity'], tag_match: 'prefix' },
    { exhaustive: true, tags: ['parity'], tag_mode: 'all' },
    { exhaustive: true, tags: ['parity'], query: 'x', exclude_tags: ['db'], current_only: false },
    { exhaustive: true, tags: ['parity'], sort: 'time_desc', min_score: 0 },
  ];
  for (const args of rejected) {
    const run = await assertParity(args);
    assert.equal(run.remoteResult.isError, true, JSON.stringify(args));
    assert.deepStrictEqual(run.remoteRequests, []);
  }
});

// --- Mode 3: ranked retrieval ----------------------------------------------------

test('ranked recall forwards exclude_tags and the state, recency and score params', async () => {
  const { remoteRequests } = await assertParity({
    query: 'database',
    tags: ['parity'],
    exclude_tags: ['deprecated', 'old'],
    current_only: false,
    state_mode: 'history',
    state_debug: true,
    recency_bias: 'on',
    min_score: 0.25,
    adaptive_floor: false,
    expand_respect_tags: true,
    scope_fallback: true,
    offset: 3,
  });
  assert.deepStrictEqual(remoteRequests, [
    'GET /recall?adaptive_floor=false&current_only=false&exclude_tags=deprecated&exclude_tags=old' +
      '&expand_respect_tags=true&min_score=0.25&offset=3&query=database&recency_bias=on' +
      '&scope_fallback=true&state_debug=true&state_mode=history&tags=parity',
  ]);
});

test('ranked recall forwards every other param the way stdio does', async () => {
  await assertParity({
    query: 'database',
    queries: ['alpha', '  ', 'beta'],
    limit: 7,
    per_query_limit: 3,
    embedding: [0.1, 0.2],
    time_query: 'last week',
    start: '2026-09-01T00:00:00Z',
    end: '2026-10-01T00:00:00Z',
    tags: ['a', 'b'],
    tag_mode: 'all',
    tag_match: 'exact',
    expand_relations: true,
    expand_entities: true,
    auto_decompose: false,
    expansion_limit: 10,
    relation_limit: 2,
    expand_min_importance: 0.4,
    expand_min_strength: 0.5,
    context: 'coding-style',
    language: 'python',
    active_path: 'src/a.py',
    context_tags: ['style'],
    context_types: ['Style'],
    priority_ids: [UUID_C],
    sort: 'time_desc',
    format: 'detailed',
  });
});

test('ranked recall drops out-of-enum values and injects no default limit', async () => {
  const { remoteRequests } = await assertParity({
    queries: ['alpha', 'beta'],
    tag_mode: 'none',
    tag_match: 'fuzzy',
    state_mode: 'bogus',
    recency_bias: 'sometimes',
  });
  assert.deepStrictEqual(remoteRequests, ['GET /recall?queries=alpha&queries=beta']);
});

test('ranked recall renders identically in every format', async () => {
  for (const format of [undefined, 'text', 'items', 'detailed', 'json']) {
    await assertParity({ query: 'parity', ...(format ? { format } : {}) });
  }
});

test('ranked recall with no results', async () => {
  await assertParity(
    { query: 'nothing', tags: ['absent'] },
    { recall: () => ({ status: 'success', results: [], count: 0, query: 'nothing' }) }
  );
});

test('ranked recall applies the same response budget', async () => {
  const many = (params) => {
    const base = richRecallResponse(params);
    const results = Array.from({ length: 20 }, (_, i) => ({
      id: `budget-${i}`,
      final_score: 1 - i / 100,
      match_type: 'vector',
      memory: {
        id: `budget-${i}`,
        content: `Budget fixture ${i}. ${'filler '.repeat(200)}`,
        tags: ['parity'],
        importance: 0.5,
        timestamp: '2026-09-01T10:00:00+00:00',
        metadata: { index: i },
      },
    }));
    return { ...base, results, count: results.length };
  };
  for (const format of ['text', 'items', 'json']) {
    const { remoteResult } = await assertParity(
      { query: 'budget', format },
      { recall: many, env: { AUTOMEM_RECALL_TOKEN_BUDGET: '2000' } }
    );
    assert.equal(remoteResult.structuredContent.truncation?.applied, true, format);
  }
});
