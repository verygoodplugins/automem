import test from "node:test";
import assert from "node:assert/strict";
import { AutoMemClient, createApp } from "../server.js";

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

// Boots the bridge against a stubbed AutoMem. `upstream(path)` returns a JSON
// body for that path, or undefined for a 404. Hands `fn` a tools/call helper
// and the list of upstream paths requested (health probes excluded).
async function withStubbedUpstream(upstream, fn) {
  const prevToken = process.env.AUTOMEM_API_TOKEN;
  const prevEndpoint = process.env.AUTOMEM_API_URL;
  process.env.AUTOMEM_API_TOKEN = "test-token";
  process.env.AUTOMEM_API_URL = "http://upstream.test";

  const originalFetch = globalThis.fetch;
  const requested = [];
  globalThis.fetch = async (url, options) => {
    const target = String(url);
    if (!target.startsWith("http://upstream.test/")) {
      return originalFetch(url, options);
    }
    const path = target.slice("http://upstream.test".length);
    if (path !== "/health") requested.push(path);
    const body = path === "/health" ? { status: "healthy" } : upstream(path);
    return new Response(JSON.stringify(body ?? { status: "error", code: 404, message: "Not Found" }), {
      status: body ? 200 : 404,
      headers: { "content-type": "application/json" },
    });
  };

  try {
    await withServer(createApp(), async (port) => {
      const post = async (path, payload) => {
        const res = await originalFetch(`http://127.0.0.1:${port}${path}`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json, text/event-stream",
            Authorization: "Bearer test-token",
          },
          body: JSON.stringify(payload),
        });
        assert.equal(res.status, 200);
        return res.json();
      };
      const callTool = async (name, args) => {
        const body = await post("/mcp", {
          jsonrpc: "2.0",
          id: 1,
          method: "tools/call",
          params: { name, arguments: args },
        });
        return body.result;
      };
      await fn({ callTool, post, requested });
    });
  } finally {
    globalThis.fetch = originalFetch;
    process.env.AUTOMEM_API_TOKEN = prevToken;
    process.env.AUTOMEM_API_URL = prevEndpoint;
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
    exclude_tags: ["deprecated", "archived"],
    expand_respect_tags: false,
    current_only: false,
    state_mode: "history",
    state_debug: true,
    recency_bias: "auto",
    min_score: 0.3,
    adaptive_floor: false,
    offset: 10,
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

  // Exclusion, current-state filtering, recency and score floors
  assert.ok(capturedPath.includes("exclude_tags=deprecated"));
  assert.ok(capturedPath.includes("exclude_tags=archived"));
  assert.ok(capturedPath.includes("expand_respect_tags=false"));
  assert.ok(capturedPath.includes("current_only=false"));
  assert.ok(capturedPath.includes("state_mode=history"));
  assert.ok(capturedPath.includes("state_debug=true"));
  assert.ok(capturedPath.includes("recency_bias=auto"));
  assert.ok(capturedPath.includes("min_score=0.3"));
  assert.ok(capturedPath.includes("adaptive_floor=false"));
  assert.ok(capturedPath.includes("offset=10"));
});

// Regression (2026-10-06): the bridge dropped arguments it did not declare, so
// recall_memory({ memory_id }) ran an unfiltered ranked search and returned five
// unrelated memories with isError false. Full parity with the stdio package is
// pinned by test/recall-parity.test.js; this is the reported symptom, end to end.
test("recall_memory memory_id fetches that one memory instead of a ranked search", async () => {
  const id = "67c0f41f-3818-48fd-8dac-af659cbb2a4f";
  await withStubbedUpstream(
    (path) =>
      path === `/memory/${id}`
        ? {
            status: "success",
            memory: { id, content: "The memory that was asked for.", tags: ["t"], timestamp: "2026-10-05T09:00:00+00:00" },
          }
        : { status: "success", results: [], count: 0 },
    async ({ callTool, requested }) => {
      const result = await callTool("recall_memory", { memory_id: id, format: "json", end: "2026-10-06T00:00:00Z" });
      assert.equal(result.isError, undefined);
      assert.deepEqual(requested, [`/memory/${id}`]);
      assert.equal(result.structuredContent.mode, "id_fetch");
      assert.deepEqual(result.structuredContent.results.map((r) => r.memory_id), [id]);
      assert.equal(result.structuredContent.results[0].content, "The memory that was asked for.");
    },
  );
});

test("recall_memory rejects a memory_id that is not a UUID before calling AutoMem", async () => {
  // Never forwarded: "by-tag" would hit GET /memory/by-tag, and ".." normalizes
  // to GET /. At UUID length, "/" and "." are still refused.
  await withStubbedUpstream(
    () => ({ status: "success", results: [], count: 0 }),
    async ({ callTool, requested }) => {
      const routeShaped = ["a".repeat(30) + "..", "a".repeat(31) + "/"];
      for (const memoryId of ["67c0f41f", "by-tag", "..", ...routeShaped]) {
        const result = await callTool("recall_memory", { memory_id: memoryId });
        assert.equal(result.isError, true, memoryId);
        assert.match(result.content[0].text, /^AutoMem error: memory_id must be a valid UUID \(request_id: /);
      }
      assert.deepEqual(requested, []);
    },
  );
});

test("recall_memory forwards every memory_id the API's uuid.UUID accepts", async () => {
  const id = "67c0f41f-3818-48fd-8dac-af659cbb2a4f";
  // uuid.UUID() also parses the rest with int(s, 16), which takes a 0x prefix,
  // underscores between digits, padding whitespace and any Unicode digit.
  const intGrammar = [
    `0x${"a".repeat(30)}`,
    `${"a".repeat(16)}_${"a".repeat(15)}`,
    `{ ${"a".repeat(30)} }`,
    `\u0661${"a".repeat(31)}`,
  ];
  const spellings = [`{${id}}`, `urn:uuid:${id}`, id.replaceAll("-", ""), id.toUpperCase(), ...intGrammar];
  await withStubbedUpstream(
    () => undefined,
    async ({ callTool, requested }) => {
      for (const memoryId of spellings) {
        const result = await callTool("recall_memory", { memory_id: memoryId });
        assert.equal(result.isError, undefined, memoryId);
        assert.equal(result.structuredContent.mode, "id_fetch");
      }
      assert.deepEqual(
        requested,
        spellings.map((memoryId) => `/memory/${encodeURIComponent(memoryId)}`),
      );
    },
  );
});

// Regression (#224): the compact block used to drop the stored date entirely, so a
// caller replaying recall text could not tell a note written today from one written
// six weeks ago, and relative language inside the content read as if it were current.
test("recall_memory text output carries the stored date on its own line", async () => {
  await withStubbedUpstream(
    () => ({
      status: "success",
      count: 1,
      results: [
        {
          id: "mem-trip",
          final_score: 0.817,
          memory: {
            id: "mem-trip",
            content: "Ground: drive up Aug 1 in Kyle's car. Kyoshk Island Aug 2-9.",
            tags: ["travel", "canada-trip"],
            timestamp: "2026-07-28T09:15:00Z",
          },
        },
      ],
    }),
    async ({ callTool }) => {
      const result = await callTool("recall_memory", { query: "trip" });
      const lines = result.content[0].text.split("\n");
      assert.ok(
        lines.some((line) => line.startsWith("   Created: 2026-07-28T09:15:00Z")),
        `expected a standalone Created line, got:\n${result.content[0].text}`,
      );
      assert.ok(lines.includes("   ID: mem-trip"));
      assert.ok(lines.some((line) => line.startsWith("1. Ground: drive up") && line.endsWith("score=0.817")));
    },
  );
});

test("recall_memory json format carries full metadata in the structured envelope", async () => {
  await withStubbedUpstream(
    () => ({
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
    }),
    async ({ callTool }) => {
      const result = await callTool("recall_memory", { query: "passthrough", format: "json" });
      const parsed = JSON.parse(result.content[0].text);
      assert.deepEqual(parsed, result.structuredContent);
      assert.equal(parsed.mode, "ranked");
      assert.deepEqual(parsed.results[0].metadata, {
        created_by: "test-agent",
        task: "synthetic-task",
      });
      assert.equal(parsed.results[0].updated_at, "2025-12-14T02:00:00Z");
      assert.equal(parsed.results[0].last_accessed, "2025-12-14T01:00:00Z");
    },
  );
});

test("Alexa RecallIntent still speaks recalled content", async () => {
  await withStubbedUpstream(
    (path) =>
      path.startsWith("/recall?")
        ? {
            status: "success",
            count: 1,
            results: [{ id: "mem-alexa", final_score: 0.8, memory: { id: "mem-alexa", content: "Feed the cat at six." } }],
          }
        : undefined,
    async ({ post }) => {
      const body = await post("/alexa", {
        request: {
          type: "IntentRequest",
          intent: { name: "RecallIntent", slots: { query: { value: "cat" } } },
        },
      });
      assert.equal(body.response.outputSpeech.text, "Item 1: Feed the cat at six.");
    },
  );
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

test("AutoMemClient.associateMemories forwards batch associations", async () => {
  const client = new AutoMemClient({
    endpoint: "http://example.test",
    apiKey: "k",
  });

  let capturedMethod = "";
  let capturedPath = "";
  let capturedBody = null;
  client._request = async (method, path, body) => {
    capturedMethod = method;
    capturedPath = path;
    capturedBody = body;
    return {
      summary: "1/2 associations created successfully",
      created_count: 1,
      failed_count: 1,
      succeeded: [{ index: 0 }],
      failed: [{ index: 1, reason: "One or both memories do not exist" }],
    };
  };

  const result = await client.associateMemories({
    associations: [
      {
        memory1_id: "11111111-1111-1111-1111-111111111111",
        memory2_id: "22222222-2222-2222-2222-222222222222",
        type: "RELATES_TO",
        strength: 0.8,
      },
      {
        memory1_id: "11111111-1111-1111-1111-111111111111",
        memory2_id: "33333333-3333-3333-3333-333333333333",
        type: "RELATES_TO",
        strength: 0.8,
      },
    ],
  });

  assert.equal(capturedMethod, "POST");
  assert.equal(capturedPath, "associate");
  assert.deepEqual(capturedBody.associations, [
    {
      memory1_id: "11111111-1111-1111-1111-111111111111",
      memory2_id: "22222222-2222-2222-2222-222222222222",
      type: "RELATES_TO",
      strength: 0.8,
    },
    {
      memory1_id: "11111111-1111-1111-1111-111111111111",
      memory2_id: "33333333-3333-3333-3333-333333333333",
      type: "RELATES_TO",
      strength: 0.8,
    },
  ]);
  assert.equal(result.message, "1/2 associations created successfully; failed index 1: One or both memories do not exist");
});

test("AutoMemClient.associateMemories caps partial failure text", async () => {
  const client = new AutoMemClient({
    endpoint: "http://example.test",
    apiKey: "k",
  });

  client._request = async () => ({
    summary: "0/12 associations created successfully",
    created_count: 0,
    failed_count: 12,
    succeeded: [],
    failed: Array.from({ length: 12 }, (_, index) => ({
      index,
      reason: `failure ${index}`,
    })),
  });

  const result = await client.associateMemories({ associations: [] });

  assert.equal(
    result.message,
    "0/12 associations created successfully; failed index 0: failure 0; failed index 1: failure 1; failed index 2: failure 2; failed index 3: failure 3; failed index 4: failure 4; 7 more failures omitted",
  );
  assert.ok(!result.message.includes("failed index 5"));
});

test("associate_memories tool returns partial success text without throwing", async () => {
  const prevToken = process.env.AUTOMEM_API_TOKEN;
  const prevEndpoint = process.env.AUTOMEM_API_URL;
  process.env.AUTOMEM_API_TOKEN = "test-token";
  process.env.AUTOMEM_API_URL = "http://upstream.test";

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    if (String(url) === "http://upstream.test/health") {
      return new Response(JSON.stringify({ status: "healthy" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    if (String(url) === "http://upstream.test/associate") {
      const body = JSON.parse(options.body);
      assert.equal(body.associations.length, 2);
      return new Response(
        JSON.stringify({
          status: "partial_success",
          summary: "1/2 associations created successfully",
          created_count: 1,
          failed_count: 1,
          succeeded: [{ index: 0 }],
          failed: [{ index: 1, reason: "One or both memories do not exist" }],
        }),
        { status: 207, headers: { "content-type": "application/json" } },
      );
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
            name: "associate_memories",
            arguments: {
              associations: [
                {
                  memory1_id: "11111111-1111-1111-1111-111111111111",
                  memory2_id: "22222222-2222-2222-2222-222222222222",
                  type: "RELATES_TO",
                  strength: 0.8,
                },
                {
                  memory1_id: "11111111-1111-1111-1111-111111111111",
                  memory2_id: "33333333-3333-3333-3333-333333333333",
                  type: "RELATES_TO",
                  strength: 0.8,
                },
              ],
            },
          },
        }),
      });

      assert.equal(res.status, 200);
      const body = await res.json();
      assert.equal(
        body.result.content[0].text,
        "1/2 associations created successfully; failed index 1: One or both memories do not exist",
      );
    });
  } finally {
    globalThis.fetch = originalFetch;
    process.env.AUTOMEM_API_TOKEN = prevToken;
    process.env.AUTOMEM_API_URL = prevEndpoint;
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
