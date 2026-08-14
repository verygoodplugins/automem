/**
 * Connects one MCP client to each transport so the two can be diffed.
 *
 * Remote: the bridge is booted in-process on an ephemeral port (same pattern as
 * test/server.test.js's withServer helper) and driven over streamable HTTP.
 * stdio: the published @verygoodplugins/mcp-automem bin is spawned as a child
 * process and driven over stdio.
 *
 * Both point at the same live AutoMem service, so any difference in the output
 * is a difference between the transports and not between two datasets.
 */
import { createRequire } from 'node:module';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { createApp } from '../server.js';

const require = createRequire(import.meta.url);

export const API_URL = process.env.AUTOMEM_PARITY_API_URL || 'http://localhost:8001';
export const API_TOKEN = process.env.AUTOMEM_PARITY_API_TOKEN || 'test-token';

export async function connectBothTransports() {
  const closers = [];

  // --- remote ---------------------------------------------------------------
  // server.js resolves the upstream inside its route handlers, so setting these
  // before createApp() is enough.
  process.env.AUTOMEM_API_URL = API_URL;
  process.env.AUTOMEM_API_TOKEN = API_TOKEN;

  const app = createApp();
  const httpServer = await new Promise((resolve) => {
    const s = app.listen(0, '127.0.0.1', () => resolve(s));
  });
  closers.push(() => new Promise((r) => httpServer.close(r)));
  const { port } = httpServer.address();

  const remote = new Client({ name: 'parity-harness', version: '1.0.0' }, {});
  const remoteTransport = new StreamableHTTPClientTransport(
    new URL(`http://127.0.0.1:${port}/mcp`),
    { requestInit: { headers: { Authorization: `Bearer ${API_TOKEN}` } } }
  );
  await remote.connect(remoteTransport);
  closers.push(() => remote.close());

  // --- stdio ----------------------------------------------------------------
  const stdioEntry = require.resolve('@verygoodplugins/mcp-automem/dist/index.js');
  const stdio = new Client({ name: 'parity-harness', version: '1.0.0' }, {});
  const stdioTransport = new StdioClientTransport({
    command: process.execPath,
    args: [stdioEntry],
    env: { ...process.env, AUTOMEM_API_URL: API_URL, AUTOMEM_API_KEY: API_TOKEN },
    stderr: 'ignore',
  });
  await stdio.connect(stdioTransport);
  closers.push(() => stdio.close());

  return {
    remote,
    stdio,
    async close() {
      for (const c of closers.reverse()) {
        await Promise.resolve(c()).catch(() => {});
      }
    },
  };
}
