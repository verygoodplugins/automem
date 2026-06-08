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

# Make the repo root importable when invoked as `python scripts/backup_automem.py`:
# Python puts the script's own dir (scripts/) on sys.path[0], not the repo root,
# so the `automem` package one level up would otherwise be unimportable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from automem.backup import (
    cleanup_old_backup_files,
    write_falkordb_backup_file,
    write_qdrant_backup_file,
)

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
    """Handles backup of AutoMem data."""

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

            backup = write_falkordb_backup_file(
                backup_dir=self.backup_dir,
                graph=graph,
                graph_name=FALKORDB_GRAPH,
                timestamp=self.timestamp,
                logger=logger,
            )
            backup_file = backup.path
            size_mb = backup_file.stat().st_size / 1024 / 1024
            logger.info(f"✅ FalkorDB backup saved: {backup_file.name} ({size_mb:.2f} MB)")
            logger.info(
                "   Nodes: %d, Relationships: %d",
                backup.stats["node_count"],
                backup.stats["relationship_count"],
            )

            return backup_file

        except Exception as e:
            logger.error(f"❌ FalkorDB backup failed: {e}")
            raise

    def backup_qdrant(self) -> Path:
        """Export Qdrant collection to JSON."""
        logger.info("🔍 Backing up Qdrant collection...")

        try:
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

            backup = write_qdrant_backup_file(
                backup_dir=self.backup_dir,
                qdrant_client=client,
                collection_name=QDRANT_COLLECTION,
                timestamp=self.timestamp,
                logger=logger,
            )
            backup_file = backup.path
            size_mb = backup_file.stat().st_size / 1024 / 1024
            logger.info(f"✅ Qdrant backup saved: {backup_file.name} ({size_mb:.2f} MB)")
            logger.info("   Points: %d", backup.stats["points_count"])

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
        cleanup_old_backup_files(backup_dir=self.backup_dir, keep=keep, logger=logger)

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
