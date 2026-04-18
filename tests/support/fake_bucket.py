"""In-memory fake for ``automem.stores.bucket_store.BucketStore``.

Keeps the test suite self-contained — no moto, no real boto3 — while
preserving the BucketStore contract used by the documents blueprint.
"""

from __future__ import annotations

import hashlib
from typing import Any, BinaryIO, Dict, Optional


class FakeBucketStore:
    """Tracks uploads in a dict keyed by bucket key.

    ``upload`` reads the entire stream (to compute sha256) and stores bytes
    verbatim. ``presigned_url`` returns a deterministic URL; tests can parse
    it to assert the key. ``delete`` removes the entry.
    """

    def __init__(self, *, bucket: str = "test-bucket") -> None:
        self.bucket = bucket
        self.objects: Dict[str, Dict[str, Any]] = {}
        # Recording knobs for assertions
        self.upload_calls = 0
        self.delete_calls = 0
        self.presign_calls = 0

    # ------------------------------------------------------------------ API
    def upload(
        self,
        key: str,
        fileobj: BinaryIO,
        *,
        mime: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        data = fileobj.read()
        size = len(data)
        sha = hashlib.sha256(data).hexdigest()
        self.objects[key] = {
            "data": data,
            "mime": mime,
            "metadata": dict(metadata or {}),
        }
        self.upload_calls += 1
        return {
            "key": key,
            "size": size,
            "sha256": sha,
            "etag": sha[:16],
            "content_type": mime,
        }

    def presigned_url(
        self,
        key: str,
        *,
        expires_in: int = 300,
        response_content_disposition: Optional[str] = None,
    ) -> str:
        self.presign_calls += 1
        qs = f"expires_in={int(expires_in)}"
        if response_content_disposition:
            # URL-safe enough for assertions; we don't actually call a server
            qs += f"&disposition={response_content_disposition.replace(' ', '+')}"
        return f"https://fake.bucket.invalid/{self.bucket}/{key}?{qs}"

    def delete(self, key: str) -> None:
        self.delete_calls += 1
        self.objects.pop(key, None)

    def head(self, key: str) -> Optional[Dict[str, Any]]:
        obj = self.objects.get(key)
        if not obj:
            return None
        return {
            "key": key,
            "size": len(obj["data"]),
            "etag": hashlib.sha256(obj["data"]).hexdigest()[:16],
            "content_type": obj["mime"],
            "last_modified": None,
            "metadata": obj["metadata"],
        }
