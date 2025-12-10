#!/usr/bin/env python3
"""Automated backup for AutoMem - FalkorDB + Qdrant.

Creates timestamped snapshots of both stores and optionally uploads to S3/cloud storage.
Designed to run as a cron job on Railway or other platforms.

Usage:
    # Local backup to ./backups/
    python scripts/backup_automem.py

    # Backup with cloud upload (requires boto3 + AWS creds)
    python scripts/backup_automem.py --s3-bucket my-automem-backups

    # Cleanup old backups (keep last N)
    python scripts/backup_automem.py --cleanup --keep 7
"""

import argparse
import gzip
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from falkordb import FalkorDB
from qdrant_client import QdrantClient

# Load environment
load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,  # Write to stdout so Railway correctly parses log levels
)
logger = logging.getLogger("automem.backup")

# Configuration
BACKUP_DIR = Path(os.getenv("AUTOMEM_BACKUP_DIR", "./backups"))
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_PASSWORD = os.getenv("FALKORDB_PASSWORD")
FALKORDB_GRAPH = os.getenv("FALKORDB_GRAPH", "memories")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "memories")


class AutoMemBackup:
    """Handles backup and restoration of AutoMem data."""

    def __init__(self, backup_dir: Path, s3_bucket: Optional[str] = None):
        self.backup_dir = backup_dir
        self.s3_bucket = s3_bucket
        self.timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Create backup directories
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        (self.backup_dir / "falkordb").mkdir(exist_ok=True)
        (self.backup_dir / "qdrant").mkdir(exist_ok=True)

    def backup_falkordb(self) -> Path:
        """Export FalkorDB graph to JSON."""
        logger.info("📊 Backing up FalkorDB graph...")

        try:
            db = FalkorDB(
                host=FALKORDB_HOST,
                port=FALKORDB_PORT,
                password=FALKORDB_PASSWORD,
                username="default" if FALKORDB_PASSWORD else None,
            )
            graph = db.select_graph(FALKORDB_GRAPH)

            # Export all nodes (using LIMIT to handle large graphs in batches)
            # Note: FalkorDB has a default result limit, so we need to paginate
            nodes = []
            batch_size = 10000
            offset = 0

            while True:
                nodes_result = graph.query(
                    f"""
                    MATCH (n)
                    RETURN
                        id(n) as id,
                        labels(n) as labels,
                        properties(n) as props
                    SKIP {offset} LIMIT {batch_size}
                """
                )

                if not nodes_result.result_set:
                    break

                batch_count = 0
                for row in nodes_result.result_set:
                    nodes.append({"id": row[0], "labels": row[1], "properties": row[2]})
                    batch_count += 1

                logger.info(f"   Exported batch: {batch_count} nodes (total: {len(nodes)})")

                if batch_count < batch_size:
                    break  # Last batch

                offset += batch_size

            # Export all relationships (using LIMIT to handle large graphs in batches)
            # Note: FalkorDB has a default result limit, so we need to paginate
            relationships = []
            batch_size = 10000
            offset = 0

            while True:
                rels_result = graph.query(
                    f"""
                    MATCH (a)-[r]->(b)
                    RETURN
                        id(a) as source_id,
                        type(r) as rel_type,
                        id(b) as target_id,
                        properties(r) as props
                    SKIP {offset} LIMIT {batch_size}
                """
                )

                if not rels_result.result_set:
                    break

                batch_count = 0
                for row in rels_result.result_set:
                    relationships.append(
                        {
                            "source_id": row[0],
                            "type": row[1],
                            "target_id": row[2],
                            "properties": row[3],
                        }
                    )
                    batch_count += 1

                logger.info(
                    f"   Exported batch: {batch_count} relationships (total: {len(relationships)})"
                )

                if batch_count < batch_size:
                    break  # Last batch

                offset += batch_size

            # Create backup data
            backup_data = {
                "timestamp": self.timestamp,
                "graph_name": FALKORDB_GRAPH,
                "nodes": nodes,
                "relationships": relationships,
                "stats": {
                    "node_count": len(nodes),
                    "relationship_count": len(relationships),
                },
            }

            # Write to compressed file
            backup_file = self.backup_dir / "falkordb" / f"falkordb_{self.timestamp}.json.gz"
            with gzip.open(backup_file, "wt", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, default=str)

            size_mb = backup_file.stat().st_size / 1024 / 1024
            logger.info(f"✅ FalkorDB backup saved: {backup_file.name} ({size_mb:.2f} MB)")
            logger.info(f"   Nodes: {len(nodes)}, Relationships: {len(relationships)}")

            return backup_file

        except Exception as e:
            logger.error(f"❌ FalkorDB backup failed: {e}")
            raise

    def backup_qdrant(self) -> Path:
        """Export Qdrant collection to JSON."""
        logger.info("🔍 Backing up Qdrant collection...")

        try:
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

            # Fetch all points
            all_points = []
            offset = None
            batch_size = 100

            while True:
                result = client.scroll(
                    collection_name=QDRANT_COLLECTION,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )

                points, next_offset = result

                for point in points:
                    all_points.append(
                        {
                            "id": point.id,
                            "vector": point.vector,
                            "payload": point.payload,
                        }
                    )

                if next_offset is None:
                    break
                offset = next_offset

            # Create backup data
            collection_info = client.get_collection(QDRANT_COLLECTION)
            backup_data = {
                "timestamp": self.timestamp,
                "collection_name": QDRANT_COLLECTION,
                "points": all_points,
                "stats": {
                    "points_count": len(all_points),
                    "vector_size": collection_info.config.params.vectors.size,
                },
            }

            # Write to compressed file
            backup_file = self.backup_dir / "qdrant" / f"qdrant_{self.timestamp}.json.gz"
            with gzip.open(backup_file, "wt", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, default=str)

            size_mb = backup_file.stat().st_size / 1024 / 1024
            logger.info(f"✅ Qdrant backup saved: {backup_file.name} ({size_mb:.2f} MB)")
            logger.info(f"   Points: {len(all_points)}")

            return backup_file

        except Exception as e:
            logger.error(f"❌ Qdrant backup failed: {e}")
            raise

    def upload_to_s3(self, file_path: Path):
        """Upload backup to S3 (requires boto3)."""
        if not self.s3_bucket:
            return

        try:
            import boto3

            s3 = boto3.client("s3")
            s3_key = f"automem-backups/{file_path.parent.name}/{file_path.name}"

            logger.info(f"☁️  Uploading to s3://{self.s3_bucket}/{s3_key}")
            s3.upload_file(str(file_path), self.s3_bucket, s3_key)
            logger.info("✅ Uploaded to S3")

        except ImportError:
            logger.warning("⚠️  boto3 not installed - skipping S3 upload")
            logger.warning("   Install with: pip install boto3")
        except Exception as e:
            logger.error(f"❌ S3 upload failed: {e}")

    def cleanup_old_backups(self, keep: int = 7):
        """Remove old backup files, keeping only the most recent N."""
        logger.info(f"🧹 Cleaning up old backups (keeping last {keep})...")

        for backup_type in ["falkordb", "qdrant"]:
            backup_path = self.backup_dir / backup_type

            # Get all backup files sorted by modification time
            backup_files = sorted(
                backup_path.glob("*.json.gz"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            # Remove old files
            for old_file in backup_files[keep:]:
                logger.info(f"   🗑️  Removing old backup: {old_file.name}")
                old_file.unlink()

            kept = min(len(backup_files), keep)
            removed = max(0, len(backup_files) - keep)
            logger.info(f"   ✅ {backup_type}: kept {kept}, removed {removed}")

    def run_backup(self, cleanup: bool = False, keep: int = 7) -> Dict[str, Any]:
        """Run full backup process."""
        logger.info(f"🚀 Starting AutoMem backup - {self.timestamp}")

        results = {
            "timestamp": self.timestamp,
            "falkordb": None,
            "qdrant": None,
            "s3_uploaded": False,
        }

        try:
            # Backup FalkorDB
            falkor_backup = self.backup_falkordb()
            results["falkordb"] = str(falkor_backup)

            if self.s3_bucket:
                self.upload_to_s3(falkor_backup)

            # Backup Qdrant
            qdrant_backup = self.backup_qdrant()
            results["qdrant"] = str(qdrant_backup)

            if self.s3_bucket:
                self.upload_to_s3(qdrant_backup)
                results["s3_uploaded"] = True

            # Cleanup old backups
            if cleanup:
                self.cleanup_old_backups(keep=keep)

            logger.info("✅ Backup completed successfully")
            return results

        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="AutoMem backup tool - exports FalkorDB and Qdrant to compressed JSON",
        epilog="""
Examples:
  # Basic backup
  python scripts/backup_automem.py

  # Backup with S3 upload
  python scripts/backup_automem.py --s3-bucket my-automem-backups

  # Backup and cleanup old files (keep last 7)
  python scripts/backup_automem.py --cleanup --keep 7

  # Custom backup directory
  python scripts/backup_automem.py --backup-dir /mnt/backups
        """,
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=str(BACKUP_DIR),
        help="Directory for backup files (default: ./backups)",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        help="S3 bucket name for cloud upload (requires boto3 and AWS credentials)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove old backups after creating new one",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=7,
        help="Number of recent backups to keep when cleaning up (default: 7)",
    )

    args = parser.parse_args()

    backup = AutoMemBackup(backup_dir=Path(args.backup_dir), s3_bucket=args.s3_bucket)

    try:
        results = backup.run_backup(cleanup=args.cleanup, keep=args.keep)
        print(json.dumps(results, indent=2))
        sys.exit(0)
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
