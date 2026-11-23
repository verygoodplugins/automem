import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { once } from 'node:events';
import { createApp } from '../server.js';

const MINUTE = 60 * 1000;

// Keep network I/O real while controlling the application's clock and periodic
// timers. This makes a one-hour session test immediate and deterministic.
async function withSseSession(t, run) {
  let now = Date.UTC(2026, 9, 20);
  const intervals = new Set();
  t.mock.method(Date, 'now', () => now);
  t.mock.method(globalThis, 'setInterval', (callback, delay) => {
    const timer = { callback, delay, unref() {} };
    intervals.add(timer);
    return timer;
  });
  t.mock.method(globalThis, 'clearInterval', (timer) => intervals.delete(timer));
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({ status: 'healthy' }), {
    headers: { 'content-type': 'application/json' },
  }));
  const savedToken = process.env.AUTOMEM_API_TOKEN;
  const savedEndpoint = process.env.AUTOMEM_API_URL;
  process.env.AUTOMEM_API_TOKEN = 'sse-test-token';
  process.env.AUTOMEM_API_URL = 'http://upstream.test';
  const app = createApp();
  let streamResponse;
  const server = http.createServer((req, res) => {
    if (req.url === '/mcp/sse') streamResponse = res;
    app(req, res);
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const port = server.address().port;
  let request;
  let response;
  try {
    const endpoint = await new Promise((resolve, reject) => {
      request = http.get(`http://127.0.0.1:${port}/mcp/sse`, (res) => {
        response = res;
        assert.equal(res.statusCode, 200);
        let buffer = '';
        res.on('data', (chunk) => {
          buffer += chunk;
          const match = buffer.match(/event: endpoint\ndata: ([^\n]+)\n\n/);
          if (match) resolve(match[1]);
        });
        res.on('error', reject);
      });
      request.on('error', reject);
    });
    const post = () => new Promise((resolve, reject) => {
      const req = http.request(`http://127.0.0.1:${port}${endpoint}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
      }, (res) => {
        res.resume();
        res.on('end', () => resolve(res.statusCode));
      });
      req.on('error', reject);
      req.end(JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'ping' }));
    });
    const sweep = (elapsed) => {
      now += elapsed;
      for (const timer of [...intervals]) {
        if (timer.delay === 5 * MINUTE) timer.callback();
      }
    };
    await run({ post, sweep, intervals, response, streamResponse });
  } finally {
    request?.destroy();
    response?.destroy();
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
    if (savedToken === undefined) delete process.env.AUTOMEM_API_TOKEN;
    else process.env.AUTOMEM_API_TOKEN = savedToken;
    if (savedEndpoint === undefined) delete process.env.AUTOMEM_API_URL;
    else process.env.AUTOMEM_API_URL = savedEndpoint;
  }
}

test('connected SSE survives the first sweep and stays usable beyond the idle TTL', async (t) => {
  await withSseSession(t, async ({ post, sweep, response }) => {
    sweep(5 * MINUTE);
    assert.equal(await post(), 202);
    assert.equal(response.destroyed, false);
    sweep(65 * MINUTE);
    assert.equal(await post(), 202);
    assert.equal(response.destroyed, false);
  });
});

test('SSE disconnect removes the session and heartbeat immediately', async (t) => {
  await withSseSession(t, async ({ post, response, streamResponse, intervals }) => {
    assert.equal([...intervals].filter((timer) => timer.delay === 20000).length, 1);
    const closed = once(streamResponse, 'close');
    response.destroy();
    await closed;
    assert.equal(await post(), 404);
    assert.equal([...intervals].filter((timer) => timer.delay === 20000).length, 0);
  });
});

test('abandoned SSE sessions expire from their last message and release heartbeat resources', async (t) => {
  await withSseSession(t, async ({ post, sweep, streamResponse, intervals }) => {
    sweep(55 * MINUTE);
    assert.equal(await post(), 202);
    // Simulate an abandoned response whose close event was missed. The real
    // HTTP stream is kept open so the sweep must close its transport itself.
    Object.defineProperty(streamResponse, 'destroyed', { configurable: true, value: true });
    sweep(10 * MINUTE);
    assert.equal(await post(), 202);
    sweep(61 * MINUTE);
    assert.equal(await post(), 404);
    assert.equal([...intervals].filter((timer) => timer.delay === 20000).length, 0);
  });
});
