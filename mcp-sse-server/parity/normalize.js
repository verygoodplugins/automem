/**
 * Normalization helpers for the cross-transport parity harness.
 *
 * Lives outside test/ on purpose: `node --test` treats every file under a
 * `test/` directory as a test file, so helper modules placed there would be
 * executed as (empty) test suites.
 */

/**
 * Recursively sort object keys so deepStrictEqual is not sensitive to the order
 * in which two independent implementations happened to build the same object.
 */
export function normalizeKeys(value) {
  if (Array.isArray(value)) return value.map(normalizeKeys);
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((k) => [k, normalizeKeys(value[k])])
    );
  }
  return value;
}

const UUID_RE = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi;
const ISO_RE = /\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})/g;
const SCORE_RE = /(score=|Score: |"final_score":\s*|"score":\s*)[\d.]+/g;
const MS_RE = /("query_time_ms":\s*)[\d.]+/g;

// Global service counters. The two transports run their scenario batches
// sequentially against one shared AutoMem, so by the time the second batch
// calls /health the first has already inserted its fixtures. These counters
// therefore always differ, and that difference is the harness's own writes —
// not a transport difference. Redact rather than reorder: no ordering makes
// a global counter agree across two sequential batches.
const VOLATILE_COUNT_KEYS = [
  'memory_count',
  'vector_count',
  'queue_depth',
  'pending',
  'inflight',
  'processed',
  'failed',
];
const JSON_COUNT_RE = new RegExp(
  `("(?:${VOLATILE_COUNT_KEYS.join('|')})":\\s*)\\d+`,
  'g'
);
const LABEL_COUNT_RE = /((?:Memory|Vector) count: )\d+/g;

/**
 * Replace values that legitimately differ run-to-run so two transports can be
 * compared on the parts that must match.
 *
 * `scopeTag` is the per-transport uuid4 tag namespace; each transport writes
 * under its own so neither sees the other's memories, which means the tag has
 * to collapse to a constant before comparison.
 */
export function redact(text, scopeTag) {
  let out = String(text);
  if (scopeTag) out = out.split(scopeTag).join('<SCOPE_TAG>');
  return out
    .replace(UUID_RE, '<UUID>')
    .replace(ISO_RE, '<TS>')
    .replace(SCORE_RE, '$1<SCORE>')
    .replace(MS_RE, '$1<MS>')
    .replace(JSON_COUNT_RE, '$1<COUNT>')
    .replace(LABEL_COUNT_RE, '$1<COUNT>');
}
