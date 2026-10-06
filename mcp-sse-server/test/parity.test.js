/**
 * Cross-transport parity: the remote bridge and the stdio package must expose
 * one identical MCP surface. See docs/MCP_TRANSPORT_PARITY.md for the contract
 * and for the short list of differences that are allowed.
 *
 * Needs a live AutoMem at :8001, so it is gated. Run it with:
 *   make test-parity
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { createRequire } from 'node:module';
import { connectBothTransports } from '../parity/clients.js';
import { normalizeKeys, redact, stripErrorFraming } from '../parity/normalize.js';
import { buildScenarios } from '../parity/scenarios.js';

const require = createRequire(import.meta.url);
const bridgePkg = require('../package.json');
const stdioPkg = require('@verygoodplugins/mcp-automem/package.json');

const UUID_RE = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/i;

/** Resolve "$PREV.<n>.memory_id" against ids already collected in this scenario. */
function resolveArgs(args, prior) {
  return JSON.parse(JSON.stringify(args), (_key, value) =>
    typeof value === 'string' && value.startsWith('$PREV.')
      ? prior[Number(value.split('.')[1])]
      : value
  );
}

/** Run every scenario against one transport, under its own tag namespace. */
async function runScenarios(client, tag) {
  const out = [];
  for (const scenario of buildScenarios(tag)) {
    const ids = [];
    const rendered = [];
    for (const call of scenario.calls) {
      const res = await client
        .callTool({ name: call.tool, arguments: resolveArgs(call.args, ids) })
        .catch((e) => ({
          content: [{ type: 'text', text: `THREW: ${e.message}` }],
          isError: true,
        }));
      const text = (res.content || []).map((c) => c.text ?? '').join('\n');
      ids.push(res.structuredContent?.memory_id ?? (text.match(UUID_RE) || [])[0]);

      // structuredContent is client-visible machine-readable output, so it is
      // part of the contract too. Comparing only text would let a mismatch in
      // memory_ids, recall count, or health statistics pass unnoticed.
      // Compared as a redacted string, not re-parsed: redaction substitutes
      // placeholders like <MS> for numeric values, so the redacted form is
      // deliberately not valid JSON. Key order is normalized first so the
      // string comparison stays meaningful.
      const structured = res.structuredContent
        ? redact(JSON.stringify(normalizeKeys(res.structuredContent)), tag)
        : null;
      rendered.push({
        tool: call.tool,
        isError: Boolean(res.isError),
        text: redact(text, tag),
        structured,
      });
    }
    out.push({ name: scenario.name, rendered });
  }
  return out;
}

const GATE =
  process.env.AUTOMEM_RUN_PARITY_TESTS === '1'
    ? false
    : 'set AUTOMEM_RUN_PARITY_TESTS=1 with a live AutoMem at :8001 (make test-parity)';

// One run of the scenario matrix per transport, shared by the tests that read
// it. A run costs ~90s, and the recall-only check reads a subset of the same
// output, so running it twice would only double the wait.
let matrixRun;
function runMatrix() {
  matrixRun ??= (async () => {
    const { remote, stdio, close } = await connectBothTransports();
    const remoteTag = `parity-remote-${randomUUID()}`;
    const stdioTag = `parity-stdio-${randomUUID()}`;
    try {
      return { a: await runScenarios(remote, remoteTag), b: await runScenarios(stdio, stdioTag) };
    } finally {
      // Every fixture carries its transport's root tag regardless of which
      // per-scenario namespace it also has, so one bulk delete per namespace is
      // enough. Bulk delete by tag exists only on the stdio transport for now,
      // so cleanup for both runs there. Swallowed so a cleanup failure can never
      // mask an assertion failure.
      await stdio
        .callTool({
          name: 'delete_memory',
          arguments: { tags: [remoteTag, stdioTag] },
        })
        .catch(() => {});
      await close();
    }
  })();
  return matrixRun;
}

/**
 * Collect every mismatch before failing. A run costs ~90s, so reporting one
 * scenario at a time turns a multi-scenario gap into a fix-one, rerun,
 * fix-the-next loop.
 */
function assertAllMatch(a, b) {
  const mismatched = [];
  for (let i = 0; i < a.length; i++) {
    try {
      assert.deepStrictEqual(a[i], b[i]);
    } catch {
      mismatched.push(i);
    }
  }
  if (mismatched.length) {
    const names = mismatched.map((i) => a[i].name).join(', ');
    const first = mismatched[0];
    assert.deepStrictEqual(
      a[first],
      b[first],
      `${mismatched.length}/${a.length} scenarios differ: ${names}\nFirst mismatch (${a[first].name}) diffed below.`
    );
  }
}

test('tools/list is identical across transports', { skip: GATE }, async () => {
  const { remote, stdio, close } = await connectBothTransports();
  try {
    const a = normalizeKeys((await remote.listTools()).tools);
    const b = normalizeKeys((await stdio.listTools()).tools);
    assert.deepStrictEqual(a, b);
  } finally {
    await close();
  }
});

test('server capabilities and instructions match', { skip: GATE }, async () => {
  const { remote, stdio, close } = await connectBothTransports();
  try {
    assert.deepStrictEqual(
      normalizeKeys(remote.getServerCapabilities()),
      normalizeKeys(stdio.getServerCapabilities())
    );
    assert.equal(remote.getInstructions(), stdio.getInstructions());

    // serverInfo is transport-specific and allowlisted to differ, but each
    // side must still report its OWN package version. The remote's hardcoded
    // 0.1.0 disagreeing with its package.json is exactly the drift the audit
    // records, so pin self-consistency rather than cross-transport equality.
    assert.equal(remote.getServerVersion().name, 'automem-mcp-sse');
    assert.equal(stdio.getServerVersion().name, 'mcp-automem');

    assert.equal(
      remote.getServerVersion().version,
      bridgePkg.version,
      'remote serverInfo.version must match mcp-sse-server/package.json'
    );
    assert.equal(
      stdio.getServerVersion().version,
      stdioPkg.version,
      'stdio serverInfo.version must match the published package version'
    );
  } finally {
    await close();
  }
});

test('tools/call renders identically across transports', { skip: GATE }, async () => {
  const { a, b } = await runMatrix();
  assertAllMatch(a, b);
});

// recall_memory is already on the shared contract (test/recall-parity.test.js
// pins it without a live stack), so it is held to parity here on its own while
// the other five tools still differ. Only the recall calls in each scenario are
// compared: the store calls that seed fixtures render differently today. Error
// framing is a transport difference; the message inside it is not.
test('recall_memory renders identically across transports', { skip: GATE }, async () => {
  const { a, b } = await runMatrix();
  const recallOnly = (runs) =>
    runs
      .map((scenario) => ({
        name: scenario.name,
        rendered: scenario.rendered
          .filter((call) => call.tool === 'recall_memory')
          .map((call) => (call.isError ? { ...call, text: stripErrorFraming(call.text) } : call)),
      }))
      .filter((scenario) => scenario.rendered.length > 0);
  assertAllMatch(recallOnly(a), recallOnly(b));
});

test('recall_memory tool definition is identical across transports', { skip: GATE }, async () => {
  const { remote, stdio, close } = await connectBothTransports();
  try {
    const pick = async (client) =>
      normalizeKeys((await client.listTools()).tools.find((tool) => tool.name === 'recall_memory'));
    assert.deepStrictEqual(await pick(remote), await pick(stdio));
  } finally {
    await close();
  }
});
