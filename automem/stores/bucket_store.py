"""S3-compatible object store for document originals.

Used by the ``/documents`` endpoints to persist uploaded file bytes alongside a
Memory node that carries the agent-supplied title + summary. Originals are
fetched lazily via short-lived presigned URLs — AutoMem never parses file
content, it just stores and hands out signed URLs to agents who know what to
do with the bytes (e.g. Claude can read PDFs natively).

Compatible with Railway Buckets, AWS S3, MinIO, Cloudflare R2, Backblaze B2,
Wasabi — anything that speaks the S3 API.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BucketConfig:
    """Immutable connection config for the bucket store."""

    endpoint: str
    bucket: str
    region: str
    access_key_id: str
    secret_access_key: str
    url_style: str = "virtual-host"
    force_path_style: bool = False


class BucketStore:
    """Thin wrapper around boto3's S3 client that exposes only the ops we use.

    Methods raise the underlying ``botocore.exceptions.ClientError`` for callers
    to classify; this keeps the wrapper dependency-free (no custom exception
    hierarchy to maintain).
    """

    def __init__(self, config: BucketConfig) -> None:
        # boto3 is an optional dependency at import-time so that test suites
        # not touching document storage don't need to install it. The client
        # is instantiated lazily on first operation.
        import boto3  # noqa: WPS433 - deliberate lazy import
        from botocore.config import Config as BotoConfig  # noqa: WPS433

        self._config = config
        addressing = (
            "path"
            if (config.force_path_style or config.url_style == "path")
            else "virtual"
        )
        self._client = boto3.client(
            "s3",
            endpoint_url=config.endpoint,
            region_name=config.region,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            config=BotoConfig(
                s3={"addressing_style": addressing},
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    # ------------------------------------------------------------------ upload
    def upload(
        self,
        key: str,
        fileobj: BinaryIO,
        *,
        mime: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Stream ``fileobj`` into the bucket at ``key``.

        The file pointer is consumed from its current position to EOF; the
        caller is responsible for positioning it (typically at 0). Returns a
        dict with ``{key, size, sha256, etag, content_type}``.
        """
        # We tee through a hashing wrapper so we capture SHA-256 without
        # loading the whole file into memory. boto3 handles chunked multipart
        # upload automatically for large files via the managed upload API.
        hasher = hashlib.sha256()
        size = 0

        class _HashingReader:
            """Wrap fileobj so boto3's managed upload streams through our hasher."""

            def __init__(self, inner: BinaryIO) -> None:
                self._inner = inner

            def read(self, n: int = -1) -> bytes:
                nonlocal size
                chunk = self._inner.read(n)
                if chunk:
                    hasher.update(chunk)
                    size += len(chunk)
                return chunk

        extra_args: Dict[str, Any] = {"ContentType": mime}
        if metadata:
            # S3 metadata keys must be ASCII, and values must be str. Let
            # boto3 URL-encode non-ASCII values via its ``Metadata`` handling.
            extra_args["Metadata"] = {
                str(k): str(v) for k, v in metadata.items() if v is not None
            }

        reader = _HashingReader(fileobj)
        self._client.upload_fileobj(
            Fileobj=reader,
            Bucket=self._config.bucket,
            Key=key,
            ExtraArgs=extra_args,
        )

        head = self._client.head_object(Bucket=self._config.bucket, Key=key)
        return {
            "key": key,
            "size": size or int(head.get("ContentLength", 0)),
            "sha256": hasher.hexdigest(),
            "etag": (head.get("ETag") or "").strip('"'),
            "content_type": head.get("ContentType", mime),
        }

    # ----------------------------------------------------------- presigned_url
    def presigned_url(
        self,
        key: str,
        *,
        expires_in: int = 300,
        response_content_disposition: Optional[str] = None,
    ) -> str:
        """Return a time-limited download URL for ``key`` (default 5 min)."""
        params: Dict[str, Any] = {"Bucket": self._config.bucket, "Key": key}
        if response_content_disposition:
            params["ResponseContentDisposition"] = response_content_disposition
        return self._client.generate_presigned_url(
            ClientMethod="get_object",
            Params=params,
            ExpiresIn=int(expires_in),
        )

    # ---------------------------------------------------------------- delete
    def delete(self, key: str) -> None:
        """Delete the object at ``key``. Idempotent: succeeds if already gone."""
        self._client.delete_object(Bucket=self._config.bucket, Key=key)

    # ----------------------------------------------------------------- head
    def head(self, key: str) -> Optional[Dict[str, Any]]:
        """Return metadata for ``key`` or None if the object does not exist."""
        try:
            response = self._client.head_object(Bucket=self._config.bucket, Key=key)
        except Exception as exc:  # pragma: no cover - boto3 error classes are dynamic
            code = getattr(getattr(exc, "response", {}), "get", lambda *_: None)(
                "Error", {}
            )
            status = (
                getattr(exc, "response", {})
                .get("ResponseMetadata", {})
                .get("HTTPStatusCode")
            )
            if code and code.get("Code") in {"404", "NoSuchKey", "NotFound"}:
                return None
            if status == 404:
                return None
            raise
        return {
            "key": key,
            "size": int(response.get("ContentLength", 0)),
            "etag": (response.get("ETag") or "").strip('"'),
            "content_type": response.get("ContentType"),
            "last_modified": response.get("LastModified"),
            "metadata": response.get("Metadata") or {},
        }


def build_bucket_store_from_config() -> Optional[BucketStore]:
    """Construct a :class:`BucketStore` from env vars, or None if unconfigured."""
    from automem.config import (
        S3_ACCESS_KEY_ID,
        S3_BUCKET,
        S3_ENDPOINT,
        S3_FORCE_PATH_STYLE,
        S3_REGION,
        S3_SECRET_ACCESS_KEY,
        S3_URL_STYLE,
        is_bucket_configured,
    )

    if not is_bucket_configured():
        return None

    assert S3_ENDPOINT and S3_BUCKET and S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY
    return BucketStore(
        BucketConfig(
            endpoint=S3_ENDPOINT,
            bucket=S3_BUCKET,
            region=S3_REGION,
            access_key_id=S3_ACCESS_KEY_ID,
            secret_access_key=S3_SECRET_ACCESS_KEY,
            url_style=S3_URL_STYLE,
            force_path_style=S3_FORCE_PATH_STYLE,
        )
    )
