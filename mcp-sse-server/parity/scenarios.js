/**
 * The tools/call scenario matrix for the parity harness.
 *
 * Each scenario runs independently against each transport, under that
 * transport's own `tag` namespace so neither sees the other's writes.
 * `$PREV.<n>.memory_id` is substituted with the memory id produced by call
 * <n> of the same scenario.
 *
 * Every scenario also owns a private `<tag>-sN` namespace and seeds its own
 * fixtures. Sharing one namespace across scenarios looks tidier but is
 * actively misleading: when a write capability differs — the remote rejects
 * `store batch` while stdio inserts three records — every later recall in the
 * shared tag compares two different datasets, so it reports a mismatch even
 * when recall rendering is already in parity. That hides which fixes actually
 * worked, exactly when you are landing them one at a time.
 *
 * Seeding therefore uses single-store only, never a mode one transport lacks.
 * Every memory also carries the root `tag`, so cleanup stays a single
 * bulk-delete per transport namespace.
 */

// Each entry is (sTag, tag) => scenario, so the private namespace is bound at
// build time and every scenario is self-contained.
const SCENARIOS = [
  (s, t) => ({
    name: 'store single',
    calls: [
      {
        tool: 'store_memory',
        args: {
          content: 'Parity fixture alpha. Chose PostgreSQL for ACID.',
          tags: [t, s],
          importance: 0.7,
        },
      },
    ],
  }),

  (s, t) => ({
    name: 'store batch',
    calls: [
      {
        tool: 'store_memory',
        args: {
          memories: [
            { content: 'Parity batch one.', tags: [t, s], importance: 0.9 },
            { content: 'Parity batch two.', tags: [t, s], importance: 0.7 },
            { content: 'Parity batch three.', tags: [t, s], importance: 0.5 },
          ],
        },
      },
    ],
  }),

  (s, t) => ({
    name: 'store supersede',
    calls: [
      {
        tool: 'store_memory',
        args: { content: 'Parity supersede original.', tags: [t, s], importance: 0.7 },
      },
      {
        tool: 'store_memory',
        args: {
          content: 'Parity supersede replacement.',
          tags: [t, s],
          importance: 0.7,
          supersedes_memory_id: '$PREV.0.memory_id',
          supersede_reason: 'parity harness',
        },
      },
    ],
  }),

  (s, t) => ({
    name: 'recall ranked text',
    calls: [
      {
        tool: 'store_memory',
        args: { content: 'Parity fixture alpha. Chose PostgreSQL for ACID.', tags: [t, s], importance: 0.9 },
      },
      {
        tool: 'store_memory',
        args: { content: 'Parity fixture beta. Qdrant holds the vectors.', tags: [t, s], importance: 0.7 },
      },
      { tool: 'recall_memory', args: { query: 'Parity fixture', tags: [s], limit: 5 } },
    ],
  }),

  (s, t) => ({
    name: 'recall detailed',
    calls: [
      {
        tool: 'store_memory',
        args: { content: 'Parity detailed fixture.', tags: [t, s], importance: 0.7 },
      },
      { tool: 'recall_memory', args: { query: 'Parity detailed', tags: [s], format: 'detailed' } },
    ],
  }),

  (s, t) => ({
    name: 'recall items',
    calls: [
      {
        tool: 'store_memory',
        args: { content: 'Parity items fixture.', tags: [t, s], importance: 0.7 },
      },
      { tool: 'recall_memory', args: { query: 'Parity items', tags: [s], format: 'items' } },
    ],
  }),

  (s, t) => ({
    name: 'recall json',
    calls: [
      {
        tool: 'store_memory',
        args: { content: 'Parity json fixture.', tags: [t, s], importance: 0.7 },
      },
      { tool: 'recall_memory', args: { query: 'Parity json', tags: [s], format: 'json' } },
    ],
  }),

  (s) => ({
    name: 'recall empty',
    calls: [
      { tool: 'recall_memory', args: { query: 'zzz-no-such-thing', tags: [`${s}-absent`] } },
    ],
  }),

  (s, t) => ({
    name: 'recall id fetch',
    calls: [
      {
        tool: 'store_memory',
        args: { content: 'Parity id-fetch target.', tags: [t, s], importance: 0.7 },
      },
      { tool: 'recall_memory', args: { memory_id: '$PREV.0.memory_id' } },
    ],
  }),

  (s, t) => ({
    // Distinct importance values are load-bearing. GET /memory/by-tag orders by
    // `importance DESC, timestamp DESC, id ASC` (automem/api/memory.py:259).
    // With equal importance and sub-millisecond writes, `id ASC` becomes the
    // tiebreaker — and ids differ between the two tag namespaces by
    // construction, so ordering would differ while actually in parity.
    name: 'recall exhaustive',
    calls: [
      { tool: 'store_memory', args: { content: 'Parity enum one.', tags: [t, s], importance: 0.9 } },
      { tool: 'store_memory', args: { content: 'Parity enum two.', tags: [t, s], importance: 0.7 } },
      { tool: 'store_memory', args: { content: 'Parity enum three.', tags: [t, s], importance: 0.5 } },
      { tool: 'recall_memory', args: { tags: [s], exhaustive: true, limit: 2, offset: 0 } },
    ],
  }),

  (s, t) => ({
    name: 'associate single',
    calls: [
      { tool: 'store_memory', args: { content: 'Parity assoc source.', tags: [t, s], importance: 0.7 } },
      { tool: 'store_memory', args: { content: 'Parity assoc target.', tags: [t, s], importance: 0.7 } },
      {
        tool: 'associate_memories',
        args: {
          memory1_id: '$PREV.0.memory_id',
          memory2_id: '$PREV.1.memory_id',
          type: 'EXEMPLIFIES',
          strength: 0.8,
          pattern_type: 'parity',
          confidence: 0.9,
        },
      },
    ],
  }),

  (s, t) => ({
    name: 'associate batch partial failure',
    calls: [
      { tool: 'store_memory', args: { content: 'Parity batch assoc source.', tags: [t, s], importance: 0.7 } },
      {
        tool: 'associate_memories',
        args: {
          associations: [
            {
              memory1_id: '$PREV.0.memory_id',
              memory2_id: '00000000-0000-4000-8000-000000000000',
              type: 'RELATES_TO',
              strength: 0.5,
            },
          ],
        },
      },
    ],
  }),

  (s, t) => ({
    name: 'update',
    calls: [
      { tool: 'store_memory', args: { content: 'Parity update original.', tags: [t, s], importance: 0.7 } },
      { tool: 'update_memory', args: { memory_id: '$PREV.0.memory_id', importance: 0.95 } },
    ],
  }),

  (s, t) => ({
    name: 'delete single',
    calls: [
      { tool: 'store_memory', args: { content: 'Parity delete target.', tags: [t, s], importance: 0.7 } },
      { tool: 'delete_memory', args: { memory_id: '$PREV.0.memory_id' } },
    ],
  }),

  (s, t) => ({
    name: 'delete by tag',
    calls: [
      { tool: 'store_memory', args: { content: 'Parity bulk delete target.', tags: [t, s], importance: 0.7 } },
      { tool: 'delete_memory', args: { tags: [s] } },
    ],
  }),

  () => ({
    name: 'health',
    calls: [{ tool: 'check_database_health', args: {} }],
  }),

  () => ({
    name: 'error: exhaustive without tags',
    calls: [{ tool: 'recall_memory', args: { exhaustive: true } }],
  }),

  (s, t) => ({
    name: 'error: store content over hard limit',
    calls: [
      { tool: 'store_memory', args: { content: 'x'.repeat(2100), tags: [t, s], importance: 0.7 } },
    ],
  }),

  () => ({
    name: 'error: unknown tool',
    calls: [{ tool: 'no_such_tool', args: {} }],
  }),
];

export function buildScenarios(tag) {
  return SCENARIOS.map((make, i) => make(`${tag}-s${i}`, tag));
}
