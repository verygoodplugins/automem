import test from "node:test";
import assert from "node:assert/strict";
import { AutoMemClient, createApp, formatRecallAsItems } from "../server.js";

async function withServer(app, fn) {
  const server = await new Promise((resolve) => {
    const s = app.listen(0, "127.0.0.1", () => resolve(s));
  });

  try {
    const address = server.address();
    assert.ok(address && typeof address === "object");
    return await fn(address.port);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
}

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
    scope_fallback: true,
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
  assert.ok(capturedPath.includes("scope_fallback=true"));

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
        updated_at: "2025-12-14T02:00:00Z",
        last_accessed: "2025-12-14T01:00:00Z",
        importance: 0.95,
        confidence: 0.88,
        type: "Insight",
        metadata: { created_by: "test-agent", task: "synthetic-task" },
      },
    },
  ];

  const detailed = formatRecallAsItems(results, { detailed: true })[0].text;
  assert.ok(detailed.includes("ID: mem-1"));
  assert.ok(detailed.includes("Type: Insight"));
  assert.ok(detailed.includes("Timestamp: 2025-12-14T00:00:00Z"));
  assert.ok(detailed.includes("Updated: 2025-12-14T02:00:00Z"));
  assert.ok(detailed.includes("Last accessed: 2025-12-14T01:00:00Z"));
  assert.ok(detailed.includes("Importance: 0.950"));
  assert.ok(detailed.includes("Confidence: 0.880"));
  assert.ok(detailed.includes("Tags: automem, cursor"));
  assert.ok(detailed.includes('Metadata: {"created_by":"test-agent","task":"synthetic-task"}'));
  assert.ok(detailed.includes("Score: 0.123"));
  assert.ok(detailed.includes("Match: relation"));
  assert.ok(detailed.includes("Source: graph"));
  assert.ok(detailed.includes("Relations: RELATES_TO(0.90) from seed-1"));

  const compact = formatRecallAsItems(results, { detailed: false })[0].text;
  assert.ok(compact.includes("score=0.123"));
  assert.ok(compact.includes("ID: mem-1"));
  assert.ok(!compact.includes("Metadata:"));
});

test("formatRecallAsItems detailed output renders full metadata and omits empty metadata", () => {
  const bigMetadata = { notes: "x".repeat(400) };
  const results = [
    {
      memory: { id: "mem-big", content: "Big metadata", metadata: bigMetadata },
    },
    {
      memory: { id: "mem-empty", content: "Empty metadata", metadata: {} },
    },
    {
      memory: { id: "mem-none", content: "No metadata" },
    },
  ];

  const [big, empty, none] = formatRecallAsItems(results, { detailed: true }).map(x => x.text);

  const metadataLine = big.split("\n").find(line => line.startsWith("Metadata: "));
  assert.ok(metadataLine, "expected a Metadata line for oversized metadata");
  const rendered = metadataLine.slice("Metadata: ".length);
  assert.equal(rendered, JSON.stringify(bigMetadata));

  assert.ok(!empty.includes("Metadata:"));
  assert.ok(!none.includes("Metadata:"));
  assert.ok(!big.includes("Updated:"));
});

test("formatRecallAsItems detailed output truncates oversized metadata previews", () => {
  const hugeMetadata = { notes: "x".repeat(5000) };
  const results = [
    {
      memory: { id: "mem-huge", content: "Huge metadata", metadata: hugeMetadata },
    },
  ];

  const [huge] = formatRecallAsItems(results, { detailed: true }).map(x => x.text);

  const metadataLine = huge.split("\n").find(line => line.startsWith("Metadata: "));
  assert.ok(metadataLine, "expected a Metadata line for huge metadata");
  const rendered = metadataLine.slice("Metadata: ".length);
  const fullJson = JSON.stringify(hugeMetadata);
  assert.ok(rendered.length < fullJson.length, "preview should be shorter than the full JSON");
  assert.ok(rendered.includes(`(truncated, ${fullJson.length} chars total)`));
});

test("recall_memory json format passes through metadata from the API response", async () => {
  const prevToken = process.env.AUTOMEM_API_TOKEN;
  const prevEndpoint = process.env.AUTOMEM_API_URL;
  process.env.AUTOMEM_API_TOKEN = "test-token";
  process.env.AUTOMEM_API_URL = "http://upstream.test";

  const originalFetch = globalThis.fetch;
  const upstreamResponse = {
    status: "success",
    results: [
      {
        id: "mem-json",
        final_score: 0.9,
        memory: {
          id: "mem-json",
          content: "JSON passthrough",
          metadata: { created_by: "test-agent", task: "synthetic-task" },
          updated_at: "2025-12-14T02:00:00Z",
          last_accessed: "2025-12-14T01:00:00Z",
        },
      },
    ],
    count: 1,
  };

  globalThis.fetch = async (url, options) => {
    if (String(url).startsWith("http://upstream.test/")) {
      return new Response(JSON.stringify(upstreamResponse), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    return originalFetch(url, options);
  };

  try {
    const app = createApp();
    await withServer(app, async (port) => {
      const res = await originalFetch(`http://127.0.0.1:${port}/mcp`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json, text/event-stream",
          Authorization: "Bearer test-token",
        },
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: 1,
          method: "tools/call",
          params: {
            name: "recall_memory",
            arguments: { query: "passthrough", format: "json" },
          },
        }),
      });

      assert.equal(res.status, 200);
      const body = await res.json();
      const text = body.result.content[0].text;
      const parsed = JSON.parse(text);
      assert.deepEqual(parsed.results[0].memory.metadata, {
        created_by: "test-agent",
        task: "synthetic-task",
      });
      assert.equal(parsed.results[0].memory.updated_at, "2025-12-14T02:00:00Z");
      assert.equal(parsed.results[0].memory.last_accessed, "2025-12-14T01:00:00Z");
    });
  } finally {
    globalThis.fetch = originalFetch;
    process.env.AUTOMEM_API_TOKEN = prevToken;
    process.env.AUTOMEM_API_URL = prevEndpoint;
  }
});

test("formatRecallAsItems surfaces outside_tag_scope fills in both formats", () => {
  const results = [
    {
      final_score: 0.42,
      outside_tag_scope: true,
      memory: { id: "fill-1", content: "Unscoped fill", tags: ["other"] },
    },
    {
      final_score: 0.9,
      memory: { id: "scoped-1", content: "Scoped result", tags: ["scoped"] },
    },
  ];

  const [fillDetailed, scopedDetailed] = formatRecallAsItems(results, { detailed: true }).map(
    (x) => x.text,
  );
  assert.ok(fillDetailed.includes("Outside tag scope: true"));
  assert.ok(!scopedDetailed.includes("Outside tag scope"));

  const [fillCompact, scopedCompact] = formatRecallAsItems(results).map((x) => x.text);
  assert.ok(fillCompact.includes("[outside tag scope]"));
  assert.ok(!scopedCompact.includes("[outside tag scope]"));
});

test("AutoMemClient._request retries transient upstream errors", async () => {
  const originalFetch = globalThis.fetch;
  const attempts = [];

  globalThis.fetch = async (_url, options) => {
    attempts.push(options?.method || "GET");
    if (attempts.length === 1) {
      return new Response(JSON.stringify({ message: "temporarily unavailable" }), {
        status: 503,
        headers: { "content-type": "application/json" },
      });
    }

    return new Response(JSON.stringify({ status: "healthy" }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  };

  try {
    const client = new AutoMemClient({
      endpoint: "http://example.test",
      apiKey: "k",
    });

    const result = await client._request("GET", "health", undefined, {
      requestId: "req-test",
      timeoutMs: 50,
      maxRetries: 1,
    });

    assert.equal(result.status, "healthy");
    assert.equal(attempts.length, 2);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

// =============================================================================
// Streamable HTTP Transport Tests (MCP 2025-03-26)
// =============================================================================

test("POST /mcp without valid JSON-RPC body returns parse error", async () => {
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
    assert.match(body.error.message, /Parse error/);
  } finally {
    await new Promise((resolve) => server.close(resolve));
    process.env.AUTOMEM_API_TOKEN = prevToken;
  }
});

test("POST /mcp with valid initialize returns stateless JSON response", async () => {
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

    const res = await fetch(`http://127.0.0.1:${port}/mcp`, {
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

    assert.equal(res.status, 200);
    assert.equal(res.headers.get("mcp-session-id"), null);
    const body = await res.json();
    assert.equal(body.result.protocolVersion, "2025-03-26");
    assert.equal(body.result.serverInfo.name, "automem-mcp-sse");
  } finally {
    await new Promise((resolve) => server.close(resolve));
    process.env.AUTOMEM_API_TOKEN = prevToken;
    process.env.AUTOMEM_API_URL = prevEndpoint;
  }
});

test("POST /mcp/ with minimal initialize returns transport-level JSON-RPC error", async () => {
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

    const res = await fetch(`http://127.0.0.1:${port}/mcp/`, {
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
      }),
    });

    assert.equal(res.status, 200);
    const body = await res.json();
    assert.ok(body.error);
    assert.equal(body.error.code, -32603);
    assert.match(body.error.message, /params/);
  } finally {
    await new Promise((resolve) => server.close(resolve));
    process.env.AUTOMEM_API_TOKEN = prevToken;
    process.env.AUTOMEM_API_URL = prevEndpoint;
  }
});

test("POST /mcp/ ignores stale session id and returns stateless JSON response", async () => {
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

    const res = await fetch(`http://127.0.0.1:${port}/mcp/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json, text/event-stream",
        Authorization: "Bearer test-token",
        "mcp-session-id": "stale-session-id",
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "tools/list",
        params: {},
      }),
    });

    assert.equal(res.status, 200);
    const body = await res.json();
    assert.ok(Array.isArray(body.result.tools));
    assert.ok(body.result.tools.length > 0);
    assert.equal(body.result.tools[0].name, "store_memory");
    assert.equal(res.headers.get("mcp-session-id"), null);
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

    const res = await fetch(`http://127.0.0.1:${port}/mcp`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // Missing Accept header
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 2,
        method: "tools/list",
        params: {},
      }),
    });

    // SDK returns 406 Not Acceptable for missing/invalid Accept header
    assert.strictEqual(res.status, 406, `Expected 406, got ${res.status}`);
    const body = await res.json();
    assert.ok(body.error, "Expected error in response body");
  } finally {
    await new Promise((resolve) => server.close(resolve));
    process.env.AUTOMEM_API_TOKEN = prevToken;
    process.env.AUTOMEM_API_URL = prevEndpoint;
  }
});

test("GET /health returns healthy when upstream is reachable", async () => {
  const originalFetch = globalThis.fetch;
  const prevToken = process.env.AUTOMEM_API_TOKEN;
  const prevEndpoint = process.env.AUTOMEM_API_URL;
  process.env.AUTOMEM_API_TOKEN = "test-token";
  process.env.AUTOMEM_API_URL = "http://example.test";

  globalThis.fetch = async (url, options) => {
    if (typeof url === "string" && url.startsWith("http://127.0.0.1:")) {
      return originalFetch(url, options);
    }

    return new Response(
      JSON.stringify({ status: "healthy", falkordb: "connected", qdrant: "connected" }),
      {
        status: 200,
        headers: { "content-type": "application/json" },
      }
    );
  };

  try {
    await withServer(createApp(), async (port) => {
      const res = await originalFetch(`http://127.0.0.1:${port}/health`);
      assert.equal(res.status, 200);

      const body = await res.json();
      assert.equal(body.status, "healthy");
      assert.equal(body.upstream, "reachable");
      assert.equal(body.upstream_details.status, "healthy");
      assert.ok(Array.isArray(body.transports));
      assert.ok(body.transports.includes("streamable-http"));
      assert.ok(body.transports.includes("sse"));
      assert.equal(body.endpoints.streamableHttp, "/mcp");
      assert.equal(body.endpoints.sse, "/mcp/sse");
      assert.ok(body.request_id);
    });
  } finally {
    globalThis.fetch = originalFetch;
    process.env.AUTOMEM_API_TOKEN = prevToken;
    process.env.AUTOMEM_API_URL = prevEndpoint;
  }
});

test("GET /health returns 200 with degraded body when upstream is unreachable", async () => {
  const originalFetch = globalThis.fetch;
  const prevToken = process.env.AUTOMEM_API_TOKEN;
  const prevEndpoint = process.env.AUTOMEM_API_URL;
  process.env.AUTOMEM_API_TOKEN = "test-token";
  process.env.AUTOMEM_API_URL = "http://example.test";

  globalThis.fetch = async (url, options) => {
    if (typeof url === "string" && url.startsWith("http://127.0.0.1:")) {
      return originalFetch(url, options);
    }

    throw new TypeError("fetch failed");
  };

  try {
    await withServer(createApp(), async (port) => {
      const res = await originalFetch(`http://127.0.0.1:${port}/health`);
      // /health is a liveness probe — always 200 while the process serves HTTP.
      // Degraded upstream is reported in the body, not via HTTP status.
      assert.equal(res.status, 200);

      const body = await res.json();
      assert.equal(body.status, "degraded");
      assert.equal(body.upstream, "unreachable");
      assert.match(body.upstream_error, /fetch failed/);
    });
  } finally {
    globalThis.fetch = originalFetch;
    process.env.AUTOMEM_API_TOKEN = prevToken;
    process.env.AUTOMEM_API_URL = prevEndpoint;
  }
});

test("GET /ready returns 503 when upstream is unreachable", async () => {
  const originalFetch = globalThis.fetch;
  const prevToken = process.env.AUTOMEM_API_TOKEN;
  const prevEndpoint = process.env.AUTOMEM_API_URL;
  process.env.AUTOMEM_API_TOKEN = "test-token";
  process.env.AUTOMEM_API_URL = "http://example.test";

  globalThis.fetch = async (url, options) => {
    if (typeof url === "string" && url.startsWith("http://127.0.0.1:")) {
      return originalFetch(url, options);
    }

    throw new TypeError("fetch failed");
  };

  try {
    await withServer(createApp(), async (port) => {
      const res = await originalFetch(`http://127.0.0.1:${port}/ready`);
      assert.equal(res.status, 503);

      const body = await res.json();
      assert.equal(body.status, "degraded");
      assert.equal(body.upstream, "unreachable");
      assert.match(body.upstream_error, /fetch failed/);
    });
  } finally {
    globalThis.fetch = originalFetch;
    process.env.AUTOMEM_API_TOKEN = prevToken;
    process.env.AUTOMEM_API_URL = prevEndpoint;
  }
});

test("GET /ready returns 200 when upstream is healthy", async () => {
  const originalFetch = globalThis.fetch;
  const prevToken = process.env.AUTOMEM_API_TOKEN;
  const prevEndpoint = process.env.AUTOMEM_API_URL;
  process.env.AUTOMEM_API_TOKEN = "test-token";
  process.env.AUTOMEM_API_URL = "http://example.test";

  globalThis.fetch = async (url, options) => {
    if (typeof url === "string" && url.startsWith("http://127.0.0.1:")) {
      return originalFetch(url, options);
    }

    return new Response(
      JSON.stringify({ status: "healthy", falkordb: "connected", qdrant: "connected" }),
      {
        status: 200,
        headers: { "content-type": "application/json" },
      }
    );
  };

  try {
    await withServer(createApp(), async (port) => {
      const res = await originalFetch(`http://127.0.0.1:${port}/ready`);
      assert.equal(res.status, 200);

      const body = await res.json();
      assert.equal(body.status, "healthy");
      assert.equal(body.upstream, "reachable");
    });
  } finally {
    globalThis.fetch = originalFetch;
    process.env.AUTOMEM_API_TOKEN = prevToken;
    process.env.AUTOMEM_API_URL = prevEndpoint;
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
