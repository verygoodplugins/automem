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
import { connectBothTransports } from '../parity/clients.js';
import { normalizeKeys } from '../parity/normalize.js';

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
