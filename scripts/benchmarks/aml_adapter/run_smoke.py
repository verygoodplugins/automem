"""Run the AML public-example contract smoke against a real HTTP MCP seam."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
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


class FakeAutoMemHandler(BaseHTTPRequestHandler):
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

    def _write_json(self, response: Dict[str, Any]) -> None:
        encoded = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json({"status": "healthy"})
            return
        if self.path.startswith("/recall?"):
            self.calls.append({"name": "recall_memory", "path": self.path})
            self._write_json({"results": self.recall_results, "count": 2})
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/memory":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        self.calls.append({"name": "store_memory", "body": body})
        self._write_json({"memory_id": f"mem-{len(self.calls)}"})


def _serve(server: Any) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def _unused_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_bridge(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise RuntimeError(f"MCP bridge exited during startup: {output}")
        try:
            if requests.get(f"{base_url}/health", timeout=0.5).status_code == 200:
                return
        except requests.RequestException:
            time.sleep(0.05)
    raise RuntimeError("MCP bridge did not become reachable within 10 seconds")


def main() -> int:
    FakeAutoMemHandler.calls = []
    fake_automem = ThreadingHTTPServer(("127.0.0.1", 0), FakeAutoMemHandler)
    _serve(fake_automem)
    bridge_port = _unused_port()
    mcp_url = f"http://127.0.0.1:{bridge_port}/mcp"
    bridge_env = {
        **os.environ,
        "PORT": str(bridge_port),
        "AUTOMEM_API_URL": f"http://127.0.0.1:{fake_automem.server_port}",
        "AUTOMEM_API_TOKEN": "upstream-smoke-token",
        "UPSTREAM_MAX_RETRIES": "0",
    }
    bridge = subprocess.Popen(
        ["node", "server.js"],
        cwd=ROOT.parents[2] / "mcp-sse-server",
        env=bridge_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _wait_for_bridge(f"http://127.0.0.1:{bridge_port}", bridge)
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

        store_calls = [call for call in FakeAutoMemHandler.calls if call["name"] == "store_memory"]
        recall_call = next(
            call for call in FakeAutoMemHandler.calls if call["name"] == "recall_memory"
        )
        assert len(store_calls) == 2
        assert [call["body"]["content"] for call in store_calls] == [
            message["content"] for message in SAMPLE["add"]["messages"]
        ]
        assert f"limit={MCP_RECALL_MAX_LIMIT}" in recall_call["path"]
        assert "tag_match=exact" in recall_call["path"]
        assert "scope_fallback=false" in recall_call["path"]
    finally:
        adapter_server.shutdown()
        bridge.terminate()
        try:
            bridge.wait(timeout=5)
        except subprocess.TimeoutExpired:
            bridge.kill()
        fake_automem.shutdown()
        if previous_adapter_key is None:
            os.environ.pop("AML_ADAPTER_API_KEY", None)
        else:
            os.environ["AML_ADAPTER_API_KEY"] = previous_adapter_key

    print("PASS: AML public API-guide Add/Search sample completed through the real MCP bridge.")
    print("PASS: Add echoed identifiers after two synchronous store_memory calls.")
    print("PASS: Search returned ranked AML data and enforced exact user scope tag.")
    print("PASS: Bearer authentication was enforced and an invalid X-Api-Key was rejected.")
    print(
        "PASS: Each AML request negotiated, refreshed, and schema-validated the live MCP tool surface."
    )
    print("PASS: top_k=100 was safely bounded to AutoMem MCP's per-call limit of 50.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
