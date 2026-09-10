"""Run the AML public-example contract smoke against a real HTTP MCP seam."""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import requests
from werkzeug.serving import make_server

# Allow the documented ``python scripts/.../run_smoke.py`` invocation as well
# as module execution, without requiring scripts/ to be a Python package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.benchmarks.aml_adapter.app import MCP_RECALL_MAX_LIMIT, McpClient, create_app

ROOT = Path(__file__).parent
SAMPLE = json.loads((ROOT / "public_sample.json").read_text())


class FakeMcpHandler(BaseHTTPRequestHandler):
    calls: List[Dict[str, Any]] = []
    recall_results: List[Dict[str, Any]] = [
        {
            "final_score": 0.98,
            "memory": {
                "id": "mem-public-1",
                "content": "Maya's preferred editor is Zed.",
                "timestamp": "2024-01-01T00:00:00Z",
            },
        },
        {
            "final_score": 0.82,
            "memory": {
                "id": "mem-public-2",
                "content": "I will use Zed in future setup instructions.",
                "timestamp": "2024-01-01T00:01:00Z",
            },
        },
    ]

    def log_message(self, _format: str, *_args: Any) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        params = body["params"]
        self.calls.append(params)
        tool = params["name"]
        if tool == "recall_memory":
            arguments = params["arguments"]
            response = {"results": self.recall_results if arguments["tags"] else [], "count": 2}
            content = [{"type": "text", "text": json.dumps(response)}]
        else:
            content = [{"type": "text", "text": "ok"}]
        output = {"jsonrpc": "2.0", "id": body["id"], "result": {"content": content}}
        encoded = json.dumps(output).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _serve(server: Any) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def main() -> int:
    FakeMcpHandler.calls = []
    fake_mcp = ThreadingHTTPServer(("127.0.0.1", 0), FakeMcpHandler)
    _serve(fake_mcp)
    mcp_url = f"http://127.0.0.1:{fake_mcp.server_port}/mcp"
    previous_adapter_key = os.environ.get("AML_ADAPTER_API_KEY")
    os.environ["AML_ADAPTER_API_KEY"] = "public-smoke-key"
    adapter_server = make_server("127.0.0.1", 0, create_app(McpClient(mcp_url)))
    _serve(adapter_server)
    base_url = f"http://127.0.0.1:{adapter_server.server_port}"
    headers = {"Authorization": "Bearer public-smoke-key"}
    try:
        add = requests.post(f"{base_url}/add", json=SAMPLE["add"], headers=headers, timeout=5)
        assert add.status_code == 200, add.text
        assert add.json() == {
            "success": True,
            "request_id": SAMPLE["add"]["request_id"],
            "user_id": SAMPLE["add"]["user_id"],
            "session_id": SAMPLE["add"]["session_id"],
        }

        search = requests.post(
            f"{base_url}/search", json=SAMPLE["search"], headers=headers, timeout=5
        )
        assert search.status_code == 200, search.text
        data = search.json()["data"]
        assert [entry["id"] for entry in data] == ["mem-public-1", "mem-public-2"]
        assert data[0]["content"] == "Maya's preferred editor is Zed."
        assert data[0]["score"] > data[1]["score"]

        unauthorized = requests.post(
            f"{base_url}/search", json=SAMPLE["search"], headers={"X-Api-Key": "wrong"}, timeout=5
        )
        assert unauthorized.status_code == 401

        store_calls = [call for call in FakeMcpHandler.calls if call["name"] == "store_memory"]
        recall_call = next(call for call in FakeMcpHandler.calls if call["name"] == "recall_memory")
        assert len(store_calls) == 2
        assert [call["arguments"]["content"] for call in store_calls] == [
            message["content"] for message in SAMPLE["add"]["messages"]
        ]
        assert recall_call["arguments"]["limit"] == MCP_RECALL_MAX_LIMIT
        assert recall_call["arguments"]["tag_match"] == "exact"
        assert recall_call["arguments"]["scope_fallback"] is False
    finally:
        adapter_server.shutdown()
        fake_mcp.shutdown()
        if previous_adapter_key is None:
            os.environ.pop("AML_ADAPTER_API_KEY", None)
        else:
            os.environ["AML_ADAPTER_API_KEY"] = previous_adapter_key

    print("PASS: AML public API-guide Add/Search sample completed over HTTP MCP tools/call.")
    print("PASS: Add echoed identifiers after two synchronous store_memory calls.")
    print("PASS: Search returned ranked AML data and enforced exact user scope tag.")
    print("PASS: Bearer authentication was enforced and an invalid X-Api-Key was rejected.")
    print("PASS: top_k=100 was safely bounded to AutoMem MCP's per-call limit of 50.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
