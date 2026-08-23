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
  const closeAll = async () => {
    for (const c of closers.reverse()) {
      await Promise.resolve(c()).catch(() => {});
    }
    closers.length = 0;
  };

  let remote;
  let stdio;

  // Setup opens an HTTP listener before the stdio child is connected. If that
  // child fails to start — the weekly @latest install resolving an incompatible
  // entry point is the realistic case — the listener would otherwise stay open,
  // and `node --test` waits on open handles, so the job would hang until
  // GitHub's timeout instead of reporting the startup failure.
  try {
    // --- remote -------------------------------------------------------------
    // server.js resolves the upstream inside its route handlers, so setting
    // these before createApp() is enough.
    process.env.AUTOMEM_API_URL = API_URL;
    process.env.AUTOMEM_API_TOKEN = API_TOKEN;

    const app = createApp();
    const httpServer = await new Promise((resolve) => {
      const s = app.listen(0, '127.0.0.1', () => resolve(s));
    });
    closers.push(() => new Promise((r) => httpServer.close(r)));
    const { port } = httpServer.address();

    remote = new Client({ name: 'parity-harness', version: '1.0.0' }, {});
    const remoteTransport = new StreamableHTTPClientTransport(
      new URL(`http://127.0.0.1:${port}/mcp`),
      { requestInit: { headers: { Authorization: `Bearer ${API_TOKEN}` } } }
    );
    await remote.connect(remoteTransport);
    closers.push(() => remote.close());

    // --- stdio --------------------------------------------------------------
    const stdioEntry = require.resolve('@verygoodplugins/mcp-automem/dist/index.js');
    stdio = new Client({ name: 'parity-harness', version: '1.0.0' }, {});
    const stdioTransport = new StdioClientTransport({
      command: process.execPath,
      args: [stdioEntry],
      env: { ...process.env, AUTOMEM_API_URL: API_URL, AUTOMEM_API_KEY: API_TOKEN },
      stderr: 'ignore',
    });
    await stdio.connect(stdioTransport);
    closers.push(() => stdio.close());
  } catch (error) {
    await closeAll();
    throw error;
  }

  return { remote, stdio, close: closeAll };
}
