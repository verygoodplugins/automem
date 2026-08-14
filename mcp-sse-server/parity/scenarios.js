/**
 * The tools/call scenario matrix for the parity harness.
 *
 * Each scenario is a short sequence of calls run independently against each
 * transport, under that transport's own `tag` namespace so neither sees the
 * other's writes. `$PREV.<n>.memory_id` is substituted with the memory id
 * produced by call <n> of the same scenario.
 */
export function buildScenarios(tag) {
  const base = { tags: [tag], importance: 0.7 };

  return [
    {
      name: 'store single',
      calls: [
        {
          tool: 'store_memory',
          args: { ...base, content: 'Parity fixture alpha. Chose PostgreSQL for ACID.' },
        },
      ],
    },
    {
      // Distinct importance values are load-bearing. GET /memory/by-tag orders by
      // `importance DESC, timestamp DESC, id ASC` (automem/api/memory.py:259). With
      // equal importance and sub-millisecond writes, `id ASC` becomes the tiebreaker
      // — and ids differ between the two tag namespaces by construction, so the
      // `recall exhaustive` scenario would fail on ordering while actually in parity.
      name: 'store batch',
      calls: [
        {
          tool: 'store_memory',
          args: {
            memories: [
              { content: 'Parity batch one.', tags: [tag], importance: 0.9 },
              { content: 'Parity batch two.', tags: [tag], importance: 0.7 },
              { content: 'Parity batch three.', tags: [tag], importance: 0.5 },
            ],
          },
        },
      ],
    },
    {
      name: 'store supersede',
      calls: [
        { tool: 'store_memory', args: { ...base, content: 'Parity supersede original.' } },
        {
          tool: 'store_memory',
          args: {
            ...base,
            content: 'Parity supersede replacement.',
            supersedes_memory_id: '$PREV.0.memory_id',
            supersede_reason: 'parity harness',
          },
        },
      ],
    },
    {
      name: 'recall ranked text',
      calls: [
        { tool: 'recall_memory', args: { query: 'Parity fixture', tags: [tag], limit: 5 } },
      ],
    },
    {
      name: 'recall detailed',
      calls: [
        {
          tool: 'recall_memory',
          args: { query: 'Parity fixture', tags: [tag], format: 'detailed' },
        },
      ],
    },
    {
      name: 'recall items',
      calls: [
        { tool: 'recall_memory', args: { query: 'Parity fixture', tags: [tag], format: 'items' } },
      ],
    },
    {
      name: 'recall json',
      calls: [
        { tool: 'recall_memory', args: { query: 'Parity fixture', tags: [tag], format: 'json' } },
      ],
    },
    {
      name: 'recall empty',
      calls: [
        {
          tool: 'recall_memory',
          args: { query: 'zzz-no-such-thing', tags: [`${tag}-absent`] },
        },
      ],
    },
    {
      name: 'recall id fetch',
      calls: [
        { tool: 'store_memory', args: { ...base, content: 'Parity id-fetch target.' } },
        { tool: 'recall_memory', args: { memory_id: '$PREV.0.memory_id' } },
      ],
    },
    {
      name: 'recall exhaustive',
      calls: [
        { tool: 'recall_memory', args: { tags: [tag], exhaustive: true, limit: 2, offset: 0 } },
      ],
    },
    {
      name: 'associate single',
      calls: [
        { tool: 'store_memory', args: { ...base, content: 'Parity assoc source.' } },
        { tool: 'store_memory', args: { ...base, content: 'Parity assoc target.' } },
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
    },
    {
      name: 'associate batch partial failure',
      calls: [
        { tool: 'store_memory', args: { ...base, content: 'Parity batch assoc source.' } },
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
    },
    {
      name: 'update',
      calls: [
        { tool: 'store_memory', args: { ...base, content: 'Parity update original.' } },
        { tool: 'update_memory', args: { memory_id: '$PREV.0.memory_id', importance: 0.95 } },
      ],
    },
    {
      name: 'delete single',
      calls: [
        { tool: 'store_memory', args: { ...base, content: 'Parity delete target.' } },
        { tool: 'delete_memory', args: { memory_id: '$PREV.0.memory_id' } },
      ],
    },
    {
      name: 'delete by tag',
      calls: [
        {
          tool: 'store_memory',
          args: { content: 'Parity bulk delete target.', tags: [`${tag}-bulk`], importance: 0.7 },
        },
        { tool: 'delete_memory', args: { tags: [`${tag}-bulk`] } },
      ],
    },
    {
      name: 'health',
      calls: [{ tool: 'check_database_health', args: {} }],
    },
    {
      name: 'error: exhaustive without tags',
      calls: [{ tool: 'recall_memory', args: { exhaustive: true } }],
    },
    {
      name: 'error: store content over hard limit',
      calls: [{ tool: 'store_memory', args: { ...base, content: 'x'.repeat(2100) } }],
    },
    {
      name: 'error: unknown tool',
      calls: [{ tool: 'no_such_tool', args: {} }],
    },
  ];
}
