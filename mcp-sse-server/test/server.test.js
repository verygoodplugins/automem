import test from "node:test";
import assert from "node:assert/strict";
import { AutoMemClient, createApp, formatRecallAsItems } from "../server.js";

test("AutoMemClient.recallMemory passes through advanced /recall params", async () => {
  const client = new AutoMemClient({
    endpoint: "http://example.test",
    apiKey: "k",
  });

  let capturedPath = "";
  client._request = async (_method, path) => {
    capturedPath = path;
    return { status: "success", results: [] };
  };

  await client.recallMemory({
    query: "hello",
    limit: 7,
    sort: "time_desc",
    tags: ["automem", "cursor"],
    tag_mode: "all",
    tag_match: "prefix",
    expand_entities: true,
    expand_relations: true,
    auto_decompose: true,
    expansion_limit: 123,
    relation_limit: 9,
    expand_min_importance: 0.6,
    expand_min_strength: 0.7,
    context: "coding-style",
    language: "python",
    active_path: "automem/api/recall.py",
    context_tags: ["style", "preferences"],
    context_types: ["Style", "Preference"],
    priority_ids: ["abc", "def"],
  });

  assert.ok(capturedPath.startsWith("recall?"));
  assert.ok(capturedPath.includes("query=hello"));
  assert.ok(capturedPath.includes("limit=7"));
  assert.ok(capturedPath.includes("sort=time_desc"));
  assert.ok(capturedPath.includes("tag_mode=all"));
  assert.ok(capturedPath.includes("tag_match=prefix"));

  assert.ok(capturedPath.includes("expand_entities=true"));
  assert.ok(capturedPath.includes("expand_relations=true"));
  assert.ok(capturedPath.includes("auto_decompose=true"));
  assert.ok(capturedPath.includes("expansion_limit=123"));
  assert.ok(capturedPath.includes("relation_limit=9"));
  assert.ok(capturedPath.includes("expand_min_importance=0.6"));
  assert.ok(capturedPath.includes("expand_min_strength=0.7"));

  assert.ok(capturedPath.includes("context=coding-style"));
  assert.ok(capturedPath.includes("language=python"));
  assert.ok(capturedPath.includes("active_path=automem%2Fapi%2Frecall.py"));

  // Arrays: repeated query params
  assert.ok(capturedPath.includes("tags=automem"));
  assert.ok(capturedPath.includes("tags=cursor"));
  assert.ok(capturedPath.includes("context_tags=style"));
  assert.ok(capturedPath.includes("context_tags=preferences"));
  assert.ok(capturedPath.includes("context_types=Style"));
  assert.ok(capturedPath.includes("context_types=Preference"));
  assert.ok(capturedPath.includes("priority_ids=abc"));
  assert.ok(capturedPath.includes("priority_ids=def"));
});

test("formatRecallAsItems supports detailed output including relations", () => {
  const results = [
    {
      final_score: 0.1234,
      match_type: "relation",
      source: "graph",
      relations: [{ type: "RELATES_TO", strength: 0.9, from: "seed-1" }],
      memory: {
        id: "mem-1",
        content: "Hello world",
        tags: ["automem", "cursor"],
        timestamp: "2025-12-14T00:00:00Z",
        last_accessed: "2025-12-14T01:00:00Z",
        importance: 0.95,
        confidence: 0.88,
        type: "Insight",
      },
    },
  ];

  const detailed = formatRecallAsItems(results, { detailed: true })[0].text;
  assert.ok(detailed.includes("ID: mem-1"));
  assert.ok(detailed.includes("Type: Insight"));
  assert.ok(detailed.includes("Timestamp: 2025-12-14T00:00:00Z"));
  assert.ok(detailed.includes("Last accessed: 2025-12-14T01:00:00Z"));
  assert.ok(detailed.includes("Importance: 0.950"));
  assert.ok(detailed.includes("Confidence: 0.880"));
  assert.ok(detailed.includes("Tags: automem, cursor"));
  assert.ok(detailed.includes("Score: 0.123"));
  assert.ok(detailed.includes("Match: relation"));
  assert.ok(detailed.includes("Source: graph"));
  assert.ok(detailed.includes("Relations: RELATES_TO(0.90) from seed-1"));

  const compact = formatRecallAsItems(results, { detailed: false })[0].text;
  assert.ok(compact.includes("score=0.123"));
  assert.ok(compact.includes("ID: mem-1"));
});

// =============================================================================
// Streamable HTTP Transport Tests (MCP 2025-03-26)
// =============================================================================

test("POST /mcp without initialize returns 400 error", async () => {
  const prevToken = process.env.AUTOMEM_API_TOKEN;
  process.env.AUTOMEM_API_TOKEN = "test-token";

  const app = createApp();
  const server = await new Promise((resolve) => {
    const s = app.listen(0, "127.0.0.1", () => resolve(s));
  });

  try {
    const address = server.address();
    const port = address.port;

    const res = await fetch(`http://127.0.0.1:${port}/mcp`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json, text/event-stream",
        Authorization: "Bearer test-token",
      },
      body: JSON.stringify({}),
    });

    assert.equal(res.status, 400);
    const body = await res.json();
    assert.ok(body.error);
    assert.ok(
      body.error.message.includes("initialize") ||
        body.error.message.includes("session")
    );
  } finally {
    await new Promise((resolve) => server.close(resolve));
    process.env.AUTOMEM_API_TOKEN = prevToken;
  }
});

test("POST /mcp with valid initialize creates session with Mcp-Session-Id", async () => {
  const prevToken = process.env.AUTOMEM_API_TOKEN;
  const prevEndpoint = process.env.AUTOMEM_API_URL;
  process.env.AUTOMEM_API_TOKEN = "test-token";
  process.env.AUTOMEM_API_URL = "http://127.0.0.1:8001";

  const app = createApp();
  const server = await new Promise((resolve) => {
    const s = app.listen(0, "127.0.0.1", () => resolve(s));
  });

  try {
    const address = server.address();
    const port = address.port;

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);

    const res = await fetch(`http://127.0.0.1:${port}/mcp`, {
      method: "POST",
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json, text/event-stream",
        Authorization: "Bearer test-token",
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "initialize",
        params: {
          protocolVersion: "2025-03-26",
          capabilities: {},
          clientInfo: { name: "test", version: "1.0" },
        },
      }),
    });

    clearTimeout(timeout);

    assert.equal(res.status, 200);
    const sessionId = res.headers.get("mcp-session-id");
    assert.ok(sessionId, "should return mcp-session-id header");
    assert.ok(sessionId.length > 10, "session ID should be a UUID");

    // Read the SSE response to get the initialize result
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (buf.length < 4096) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value) buf += decoder.decode(value, { stream: true });
      if (buf.includes('"protocolVersion"')) break;
    }
    await reader.cancel();

    assert.ok(buf.includes("event: message"), "should return SSE event");
    assert.ok(
      buf.includes('"protocolVersion"'),
      "should return initialize result"
    );
  } finally {
    await new Promise((resolve) => server.close(resolve));
    process.env.AUTOMEM_API_TOKEN = prevToken;
    process.env.AUTOMEM_API_URL = prevEndpoint;
  }
});

test("POST /mcp without Accept header returns error", async () => {
  const prevToken = process.env.AUTOMEM_API_TOKEN;
  const prevEndpoint = process.env.AUTOMEM_API_URL;
  process.env.AUTOMEM_API_TOKEN = "test-token";
  process.env.AUTOMEM_API_URL = "http://127.0.0.1:8001";

  const app = createApp();
  const server = await new Promise((resolve) => {
    const s = app.listen(0, "127.0.0.1", () => resolve(s));
  });

  try {
    const address = server.address();
    const port = address.port;

    // First create a valid session
    const initRes = await fetch(`http://127.0.0.1:${port}/mcp`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json, text/event-stream",
        Authorization: "Bearer test-token",
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "initialize",
        params: {
          protocolVersion: "2025-03-26",
          capabilities: {},
          clientInfo: { name: "test", version: "1.0" },
        },
      }),
    });

    const sessionId = initRes.headers.get("mcp-session-id");
    // Consume the init response
    await initRes.text();

    // Now try to use it without Accept header
    const res = await fetch(`http://127.0.0.1:${port}/mcp`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "mcp-session-id": sessionId,
        // Missing Accept header
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 2,
        method: "tools/list",
        params: {},
      }),
    });

    // SDK returns 4xx when Accept header doesn't include required types
    // Behavior changed in SDK 1.20+: returns 400 instead of 406
    assert.ok(
      [400, 406].includes(res.status),
      `Expected 400 or 406, got ${res.status}`
    );
    const body = await res.json();
    assert.ok(body.error, "Expected error in response body");
  } finally {
    await new Promise((resolve) => server.close(resolve));
    process.env.AUTOMEM_API_TOKEN = prevToken;
    process.env.AUTOMEM_API_URL = prevEndpoint;
  }
});

test("GET /health returns both transport types", async () => {
  const app = createApp();
  const server = await new Promise((resolve) => {
    const s = app.listen(0, "127.0.0.1", () => resolve(s));
  });

  try {
    const address = server.address();
    const port = address.port;

    const res = await fetch(`http://127.0.0.1:${port}/health`);
    assert.equal(res.status, 200);

    const body = await res.json();
    assert.equal(body.status, "healthy");
    assert.ok(Array.isArray(body.transports));
    assert.ok(body.transports.includes("streamable-http"));
    assert.ok(body.transports.includes("sse"));
    assert.ok(body.endpoints);
    assert.equal(body.endpoints.streamableHttp, "/mcp");
    assert.equal(body.endpoints.sse, "/mcp/sse");
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

// =============================================================================
// SSE Transport Tests (MCP 2024-11-05 - Deprecated)
// =============================================================================

test("GET /mcp/sse returns an SSE stream and endpoint event", async () => {
  const prevToken = process.env.AUTOMEM_API_TOKEN;
  const prevEndpoint = process.env.AUTOMEM_API_URL;
  process.env.AUTOMEM_API_TOKEN = "test-token";
  process.env.AUTOMEM_API_URL = "http://127.0.0.1:8001";

  const app = createApp();
  const server = await new Promise((resolve) => {
    const s = app.listen(0, "127.0.0.1", () => resolve(s));
  });

  try {
    const address = server.address();
    assert.ok(address && typeof address === "object");
    const port = address.port;

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1500);

    const res = await fetch(`http://127.0.0.1:${port}/mcp/sse`, {
      signal: controller.signal,
      headers: { Accept: "text/event-stream" },
    });

    assert.equal(res.status, 200);
    const ct = res.headers.get("content-type") || "";
    assert.ok(ct.includes("text/event-stream"));
    assert.ok(res.body);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (buf.length < 8192) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value) buf += decoder.decode(value, { stream: true });
      if (buf.includes("event: endpoint")) break;
    }

    clearTimeout(timeout);
    await reader.cancel();

    assert.ok(
      buf.includes("event: endpoint"),
      `missing endpoint event; got:\n${buf}`
    );
  } finally {
    await new Promise((resolve) => server.close(resolve));
    process.env.AUTOMEM_API_TOKEN = prevToken;
    process.env.AUTOMEM_API_URL = prevEndpoint;
  }
});
