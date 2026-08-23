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
import { connectBothTransports } from '../parity/clients.js';
import { normalizeKeys, redact } from '../parity/normalize.js';
import { buildScenarios } from '../parity/scenarios.js';

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
      const structured = res.structuredContent
        ? JSON.parse(redact(JSON.stringify(normalizeKeys(res.structuredContent)), tag))
        : null;
      rendered.push({ isError: Boolean(res.isError), text: redact(text, tag), structured });
    }
    out.push({ name: scenario.name, rendered });
  }
  return out;
}

const GATE =
  process.env.AUTOMEM_RUN_PARITY_TESTS === '1'
    ? false
    : 'set AUTOMEM_RUN_PARITY_TESTS=1 with a live AutoMem at :8001 (make test-parity)';

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

    // serverInfo.name is an allowlisted difference — clients must be able to
    // tell the transports apart (docs/MCP_TRANSPORT_PARITY.md).
    assert.equal(remote.getServerVersion().name, 'automem-mcp-sse');
    assert.equal(stdio.getServerVersion().name, 'mcp-automem');
  } finally {
    await close();
  }
});

test('tools/call renders identically across transports', { skip: GATE }, async () => {
  const { remote, stdio, close } = await connectBothTransports();
  const remoteTag = `parity-remote-${randomUUID()}`;
  const stdioTag = `parity-stdio-${randomUUID()}`;
  try {
    const a = await runScenarios(remote, remoteTag);
    const b = await runScenarios(stdio, stdioTag);

    // Collect every mismatch before failing. A run costs ~90s, so reporting
    // one scenario at a time turns a multi-scenario gap into a fix-one,
    // rerun, fix-the-next loop.
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
  } finally {
    // Bulk delete by tag exists only on the stdio transport for now, so cleanup
    // for both namespaces runs there. Swallowed so a cleanup failure can never
    // mask an assertion failure.
    await stdio
      .callTool({
        name: 'delete_memory',
        arguments: {
          tags: [remoteTag, stdioTag, `${remoteTag}-bulk`, `${stdioTag}-bulk`],
        },
      })
      .catch(() => {});
    await close();
  }
});
