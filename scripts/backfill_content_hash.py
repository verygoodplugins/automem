"""Backfill the Qdrant `content_hash` payload field on points written before #270.

Optional: enrichment stamps a legacy point lazily the next time it enriches it, so
this script is only a bulk option for stamping every point at once. Each write is
filtered on `content_hash` still being empty: every writer now sets the field, so a
point that is still empty has not been PATCHed since it was scrolled, and the
backfill cannot stamp an old hash on new content.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from automem.stores.vector_payload import content_fingerprint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--apply", action="store_true", help="Write the field (default is a dry run)."
    )
    args = parser.parse_args()

    from qdrant_client import QdrantClient, models

    from automem.config import COLLECTION_NAME, QDRANT_API_KEY, QDRANT_URL

    # automem.config builds QDRANT_URL from QDRANT_HOST + QDRANT_PORT (Railway) when unset.
    url = (QDRANT_URL or "").strip()
    if not url:
        print("QDRANT_URL or QDRANT_HOST is required", file=sys.stderr)
        return 1
    client = QdrantClient(url=url, api_key=(QDRANT_API_KEY or "").strip() or None)
    collection = (COLLECTION_NAME or "memories").strip()
    missing = models.IsEmptyCondition(is_empty=models.PayloadField(key="content_hash"))

    scanned = updated = failures = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=[missing]),
            limit=args.batch_size,
            offset=offset,
            with_payload=["content"],
            with_vectors=False,
        )
        for point in points:
            scanned += 1
            if not args.apply:
                continue
            try:
                client.set_payload(
                    collection_name=collection,
                    points=models.Filter(must=[models.HasIdCondition(has_id=[point.id]), missing]),
                    payload={
                        "content_hash": content_fingerprint((point.payload or {}).get("content"))
                    },
                    wait=True,
                )
                updated += 1
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures += 1
                print(f"failed {point.id}: {exc}", file=sys.stderr)
        if offset is None:
            break

    print(
        json.dumps(
            {
                "scanned": scanned,
                "updated": updated,
                "failures": failures,
                "dry_run": not args.apply,
            }
        )
    )
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
