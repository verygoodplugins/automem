/**
 * A canned AutoMem API for the recall parity guard (test/recall-parity.test.js).
 *
 * It answers the three routes recall_memory uses with the shapes automem/api
 * returns, so the remote bridge and the published stdio package can be driven
 * through identical upstream responses without a live stack. Every request is
 * recorded, which is how the guard checks that both clients ask the API the
 * same thing — not only that they render the same answer.
 *
 * Lives outside test/ because `node --test` runs every file under test/.
 */

export const FAKE_API_URL = 'http://automem.parity.test';

export const UUID_A = '11111111-1111-4111-8111-111111111111';
export const UUID_B = '22222222-2222-4222-8222-222222222222';
export const UUID_C = '33333333-3333-4333-8333-333333333333';
export const UUID_D = '44444444-4444-4444-8444-444444444444';
export const UUID_MISSING = '00000000-0000-4000-8000-000000000000';

// Stored node shapes, as GET /memory/<id> and GET /memory/by-tag return them.
export const MEMORIES = [
  {
    id: UUID_A,
    content: 'Parity fixture alpha. Chose PostgreSQL for ACID.',
    summary: 'Chose PostgreSQL.',
    tags: ['parity', 'db'],
    importance: 0.9,
    confidence: 0.95,
    type: 'Decision',
    timestamp: '2026-09-01T10:00:00+00:00',
    updated_at: '2026-09-02T11:00:00+00:00',
    last_accessed: '2026-10-01T12:00:00+00:00',
    metadata: { source: 'parity', files: ['a.py'] },
  },
  {
    // Longer than the 400-char preview, and no updated_at, so the
    // updated_at -> timestamp fallback is exercised.
    id: UUID_B,
    content: `Parity fixture beta. ${'Qdrant holds the vectors. '.repeat(24)}`.trim(),
    tags: ['parity'],
    importance: 0.7,
    type: 'Context',
    timestamp: '2026-09-03T10:00:00+00:00',
    metadata: {},
  },
  {
    // Empty content: the text channel falls back to the stored summary.
    id: UUID_C,
    content: '',
    summary: 'Summary-only fixture.',
    tags: ['parity', 'other'],
    importance: 0.5,
    timestamp: '2026-09-04T10:00:00+00:00',
    updated_at: '2026-09-05T10:00:00+00:00',
  },
  {
    id: UUID_D,
    content: 'Unrelated fixture.',
    tags: ['unrelated'],
    importance: 0.3,
    timestamp: '2026-09-06T10:00:00+00:00',
  },
];

// Python's uuid.UUID(), which GET /memory/<id> uses to validate: it accepts
// braces, a urn:uuid: prefix and missing hyphens, and rejects everything else.
function isPythonUuid(value) {
  const hex = value
    .replaceAll('urn:', '')
    .replaceAll('uuid:', '')
    .replace(/^[{}]+|[{}]+$/g, '')
    .replaceAll('-', '');
  return /^[0-9a-f]{32}$/i.test(hex);
}

function byTagPage(params) {
  const tags = params
    .getAll('tags')
    .map((t) => t.trim().toLowerCase())
    .filter(Boolean);
  if (!tags.length) {
    return [400, { status: 'error', code: 400, message: "'tags' query parameter is required" }];
  }
  const limit = Math.max(1, Math.min(Number.parseInt(params.get('limit') ?? '20', 10) || 20, 200));
  const offset = Math.max(0, Number.parseInt(params.get('offset') ?? '0', 10) || 0);
  // Same ordering as automem/api/memory.py: importance DESC, timestamp DESC, id ASC.
  const matching = MEMORIES.filter((m) => m.tags.some((t) => tags.includes(t.toLowerCase()))).sort(
    (x, y) =>
      y.importance - x.importance ||
      y.timestamp.localeCompare(x.timestamp) ||
      x.id.localeCompare(y.id)
  );
  const memories = matching.slice(offset, offset + limit);
  return [
    200,
    {
      status: 'success',
      tags,
      count: memories.length,
      limit,
      offset,
      has_more: matching.length > offset + limit,
      memories,
    },
  ];
}

/**
 * The default /recall answer: three scored hits carrying every result-level
 * field the renderers read, plus a fully populated envelope.
 */
export function richRecallResponse(params) {
  const [a, b, c, d] = MEMORIES;
  const relation = (memory, type, strength) => ({ type, strength, memory });
  const excludeTags = params.getAll('exclude_tags');
  return {
    status: 'success',
    results: [
      {
        id: a.id,
        final_score: 0.91234,
        match_type: 'vector',
        match_score: 0.8,
        relation_score: 0.1,
        source: 'qdrant',
        score_components: { vector: 0.5, keyword: 0.3 },
        jit_enriched: true,
        relations: [
          relation(b, 'RELATES_TO', 0.8),
          relation(c, 'LEADS_TO', 0.6),
          relation(d, 'PART_OF', 0.4),
          relation({ id: 'rel-4', content: 'x'.repeat(150) }, 'EXEMPLIFIES', 0.3),
        ],
        memory: a,
      },
      {
        id: b.id,
        final_score: 0.7,
        match_type: 'keyword',
        deduped_from: ['dup-1', 'dup-2'],
        state_replaces: d.id,
        memory: b,
      },
      {
        id: c.id,
        score: 0.42,
        match_type: 'entity',
        expanded_from_entity: 'postgresql',
        outside_tag_scope: true,
        related_to: [relation(a, 'RELATES_TO', 0.9)],
        memory: c,
      },
    ],
    count: 3,
    dedup_removed: 2,
    query: params.get('query') ?? '',
    sort: params.get('sort') ?? 'score',
    keywords: ['parity', 'database'],
    time_window: { start: null, end: null },
    tags: params.getAll('tags'),
    ...(excludeTags.length ? { exclude_tags: excludeTags } : {}),
    tag_mode: params.get('tag_mode') ?? 'any',
    tag_match: params.get('tag_match') ?? 'prefix',
    state_mode: params.get('state_mode') ?? 'current',
    tag_scope: { filtered: true, gated_low_evidence: 0 },
    scope_fallback: params.get('scope_fallback') === 'true',
    recency_bias: params.get('recency_bias') ?? 'off',
    score_filter: {
      min_score: Number(params.get('min_score') ?? 0),
      adaptive_floor: params.get('adaptive_floor') !== 'false',
      filtered_count: 2,
    },
    queries: params.getAll('queries'),
    vector_search: { enabled: true, matched: true },
    jit_enriched_count: 1,
    query_time_ms: 12.5,
    entities: [{ name: 'PostgreSQL', type: 'tool' }],
    expansion: { enabled: true, expanded_count: 1 },
    entity_expansion: { enabled: true, expanded_count: 1, entities_found: ['postgresql'] },
    context_priority: { context: params.get('context') ?? null },
    state_filter: { suppressed_count: 1, replacement_count: 1 },
  };
}

export function createFakeAutoMem() {
  let requests = [];
  let recallResponder = richRecallResponse;

  function respond(status, body) {
    return new Response(JSON.stringify(body), {
      status,
      headers: { 'content-type': 'application/json' },
    });
  }

  async function handle(url, init = {}) {
    const parsed = new URL(url);
    const method = (init.method || 'GET').toUpperCase();
    const { pathname, searchParams } = parsed;

    if (pathname === '/health') {
      return respond(200, { status: 'healthy' });
    }

    // Key order is not part of the contract; same-key order (repeated tags) is.
    const query = [...searchParams]
      .sort(([x], [y]) => (x < y ? -1 : x > y ? 1 : 0))
      .map(([k, v]) => `${k}=${v}`)
      .join('&');
    requests.push(`${method} ${pathname}${query ? `?${query}` : ''}`);

    if (method === 'GET' && pathname === '/recall') {
      return respond(200, recallResponder(searchParams));
    }
    if (method === 'GET' && pathname === '/memory/by-tag') {
      return respond(...byTagPage(searchParams));
    }
    if (method === 'GET' && pathname.startsWith('/memory/')) {
      const id = decodeURIComponent(pathname.slice('/memory/'.length));
      if (!isPythonUuid(id)) {
        return respond(400, { status: 'error', code: 400, message: 'memory_id must be a valid UUID' });
      }
      const memory = MEMORIES.find((m) => m.id === id);
      return memory
        ? respond(200, { status: 'success', memory })
        : respond(404, { status: 'error', code: 404, message: 'Memory not found' });
    }
    return respond(404, { status: 'error', code: 404, message: 'Not Found' });
  }

  return {
    handle,
    /** Returns the requests recorded since the last call, then clears them. */
    takeRequests() {
      const taken = requests;
      requests = [];
      return taken;
    },
    /** Replace the /recall answer; omit to restore the default. */
    setRecallResponder(fn) {
      recallResponder = fn || richRecallResponse;
    },
  };
}
