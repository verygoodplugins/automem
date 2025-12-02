#!/usr/bin/env python3
"""Restore AutoMem from backup files.

Restores FalkorDB graph and Qdrant vectors from compressed JSON backups.

Usage:
    # Restore from latest backup
    python scripts/restore_from_backup.py

    # Restore from specific backup
    python scripts/restore_from_backup.py --backup-timestamp 20251019_085625

    # Dry run (show what would be restored)
    python scripts/restore_from_backup.py --dry-run

    # Import/merge without deleting existing data
    python scripts/restore_from_backup.py --import
"""

import argparse
import gzip
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from falkordb import FalkorDB
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Load environment
load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("automem.restore")

# Configuration
BACKUP_DIR = Path(os.getenv("AUTOMEM_BACKUP_DIR", "./backups"))
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_PASSWORD = os.getenv("FALKORDB_PASSWORD")
FALKORDB_GRAPH = os.getenv("FALKORDB_GRAPH", "memories")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "memories")


class AutoMemRestore:
    """Handles restoration of AutoMem data from backups."""

    def __init__(self, backup_dir: Path, dry_run: bool = False, force: bool = False, merge: bool = False):
        self.backup_dir = backup_dir
        self.dry_run = dry_run
        self.force = force
        self.merge = merge

    def find_latest_backup(self, backup_type: str) -> Path:
        """Find the most recent backup file."""
        backup_path = self.backup_dir / backup_type
        backup_files = sorted(
            backup_path.glob("*.json.gz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not backup_files:
            raise FileNotFoundError(f"No {backup_type} backups found in {backup_path}")

        return backup_files[0]

    def find_backup_by_timestamp(self, backup_type: str, timestamp: str) -> Path:
        """Find backup file by timestamp."""
        backup_file = self.backup_dir / backup_type / f"{backup_type}_{timestamp}.json.gz"
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup not found: {backup_file}")
        return backup_file

    def restore_falkordb(self, backup_file: Path) -> Dict[str, Any]:
        """Restore FalkorDB graph from JSON backup."""
        logger.info(f"📊 Restoring FalkorDB from {backup_file.name}...")

        # Load backup data
        with gzip.open(backup_file, "rt", encoding="utf-8") as f:
            backup_data = json.load(f)

        logger.info(f"   Backup contains {len(backup_data['nodes'])} nodes, "
                   f"{len(backup_data['relationships'])} relationships")

        if self.dry_run:
            logger.info("   [DRY RUN] Would restore to FalkorDB")
            return {
                "nodes": len(backup_data['nodes']),
                "relationships": len(backup_data['relationships']),
                "dry_run": True
            }

        # Connect to FalkorDB
        db = FalkorDB(
            host=FALKORDB_HOST,
            port=FALKORDB_PORT,
            password=FALKORDB_PASSWORD,
            username="default" if FALKORDB_PASSWORD else None
        )
        graph = db.select_graph(FALKORDB_GRAPH)

        # Warning about existing data
        existing_count = graph.query("MATCH (n) RETURN count(*) as count")
        existing_nodes = existing_count.result_set[0][0] if existing_count.result_set else 0
        
        if existing_nodes > 0:
            if self.merge:
                logger.info(f"📥 Import mode: Graph '{FALKORDB_GRAPH}' contains {existing_nodes} existing nodes - will merge with backup")
            else:
                logger.warning(f"⚠️  Graph '{FALKORDB_GRAPH}' contains {existing_nodes} existing nodes!")
                if not self.force:
                    response = input("   Delete existing data and restore? [y/N]: ")
                    if response.lower() != 'y':
                        logger.info("   Restore cancelled")
                        return {"cancelled": True}
                else:
                    logger.info("   --force flag set, proceeding with restore")

                # Clear existing data
                logger.info("   🗑️  Clearing existing graph data...")
                graph.query("MATCH (n) DETACH DELETE n")

        # Restore nodes
        logger.info(f"   📥 Restoring {len(backup_data['nodes'])} nodes...")
        node_backup_id_to_props = {}  # Map backup IDs to properties
        nodes_created = 0
        nodes_skipped = 0
        restore_time = datetime.now(timezone.utc).isoformat()

        # If merging, get existing node UUIDs to skip duplicates
        existing_uuids = set()
        if self.merge:
            existing_nodes_result = graph.query("MATCH (n) WHERE n.id IS NOT NULL RETURN n.id as id")
            if existing_nodes_result.result_set:
                existing_uuids = {row[0] for row in existing_nodes_result.result_set}

        for i, node in enumerate(backup_data['nodes']):
            if i % 100 == 0:
                logger.info(f"      Progress: {i}/{len(backup_data['nodes'])}")

            labels = ':'.join(node['labels'])
            props = node['properties'].copy()
            
            # Check if node already exists (by UUID)
            node_uuid = props.get('id')
            if self.merge and node_uuid and node_uuid in existing_uuids:
                nodes_skipped += 1
                node_backup_id_to_props[node['id']] = (labels, props)
                continue
            
            # For Memory nodes, set default relevance_score and last_accessed to prevent
            # immediate deletion by consolidation scheduler
            if 'Memory' in node['labels']:
                # Set last_accessed to now so memories are treated as recently accessed
                props['last_accessed'] = restore_time
                
                # Set initial relevance_score based on importance, or default to 0.5
                # This prevents old memories from being immediately deleted
                if 'relevance_score' not in props or props.get('relevance_score') is None:
                    importance = props.get('importance', 0.5) or 0.5
                    # Base relevance on importance, but ensure minimum of 0.3 to prevent deletion
                    props['relevance_score'] = max(0.3, float(importance))
            
            node_backup_id_to_props[node['id']] = (labels, props)

            # Convert properties to Cypher format - escape property names
            props_list = []
            for k, v in props.items():
                # Use backticks for property names to handle special characters
                props_list.append(f"`{k}`: {json.dumps(v)}")
            props_str = ', '.join(props_list)

            # Create node with escaped labels
            escaped_labels = ':'.join(f"`{label}`" if ' ' in label else label for label in node['labels'])
            query = f"CREATE (n:{escaped_labels} {{{props_str}}})"
            try:
                result = graph.query(query)
                nodes_created += 1
            except Exception as e:
                if i < 5 or i % 100 == 0:  # Log first 5 and every 100th error
                    logger.warning(f"      Error creating node {i}: {e}")
                continue

        logger.info(f"   ✅ Restored {nodes_created}/{len(backup_data['nodes'])} nodes" + 
                   (f" (skipped {nodes_skipped} existing)" if nodes_skipped > 0 else ""))

        # Restore relationships using UUID matching
        logger.info(f"   📥 Restoring {len(backup_data['relationships'])} relationships...")
        rel_created = 0
        rel_skipped = 0

        # If merging, get existing relationships to skip duplicates
        existing_rels = set()
        if self.merge:
            existing_rels_result = graph.query("""
                MATCH (a)-[r]->(b)
                WHERE a.id IS NOT NULL AND b.id IS NOT NULL
                RETURN type(r) as rel_type, a.id as source_id, b.id as target_id
            """)
            if existing_rels_result.result_set:
                existing_rels = {(row[0], row[1], row[2]) for row in existing_rels_result.result_set}

        for i, rel in enumerate(backup_data['relationships']):
            if i % 100 == 0:
                logger.info(f"      Progress: {i}/{len(backup_data['relationships'])}")

            # Get source and target node IDs from backup
            source_backup_id = rel['source_id']
            target_backup_id = rel['target_id']

            if source_backup_id not in node_backup_id_to_props or target_backup_id not in node_backup_id_to_props:
                logger.warning(f"      Skipping relationship {rel['type']} - missing node IDs")
                continue

            source_labels, source_props = node_backup_id_to_props[source_backup_id]
            target_labels, target_props = node_backup_id_to_props[target_backup_id]

            # Get the unique ID from the source node properties
            source_uuid = source_props.get('id')
            target_uuid = target_props.get('id')

            if not source_uuid or not target_uuid:
                logger.warning(f"      Skipping relationship {rel['type']} - missing UUID properties")
                continue

            rel_type = rel['type']
            
            # Check if relationship already exists
            if self.merge and (rel_type, source_uuid, target_uuid) in existing_rels:
                rel_skipped += 1
                continue
            
            props = rel.get('properties', {})
            
            # Build relationship properties string
            props_list = []
            for k, v in props.items():
                props_list.append(f"`{k}`: {json.dumps(v)}")
            props_str = ', '.join(props_list)
            if props_str:
                rel_props_str = f" {{{props_str}}}"
            else:
                rel_props_str = ""

            # Use UUID to match nodes - more reliable than internal IDs
            first_source_label = source_labels.split(':')[0].strip('`')
            first_target_label = target_labels.split(':')[0].strip('`')
            escaped_rel_type = f"`{rel_type}`" if ' ' in rel_type else rel_type
            
            # Use MERGE in import mode to handle duplicates gracefully
            if self.merge:
                query = f"""
                    MATCH (a:{first_source_label} {{id: "{source_uuid}"}}),
                          (b:{first_target_label} {{id: "{target_uuid}"}})
                    MERGE (a)-[r:{escaped_rel_type}{rel_props_str}]->(b)
                """
            else:
                query = f"""
                    MATCH (a:{first_source_label} {{id: "{source_uuid}"}}),
                          (b:{first_target_label} {{id: "{target_uuid}"}})
                    CREATE (a)-[r:{escaped_rel_type}{rel_props_str}]->(b)
                """
            try:
                graph.query(query)
                rel_created += 1
            except Exception as e:
                logger.debug(f"      Skipped relationship {rel['type']}: {e}")
                continue

        logger.info(f"   ✅ Restored {rel_created}/{len(backup_data['relationships'])} relationships" +
                   (f" (skipped {rel_skipped} existing)" if rel_skipped > 0 else ""))

        return {
            "nodes_restored": nodes_created,
            "nodes_skipped": nodes_skipped if self.merge else 0,
            "nodes_attempted": len(backup_data['nodes']),
            "relationships_restored": rel_created,
            "relationships_skipped": rel_skipped if self.merge else 0,
            "relationships_attempted": len(backup_data['relationships']),
            "dry_run": False,
            "merge_mode": self.merge
        }

    def restore_qdrant(self, backup_file: Path) -> Dict[str, Any]:
        """Restore Qdrant collection from JSON backup."""
        logger.info(f"🔍 Restoring Qdrant from {backup_file.name}...")

        # Load backup data
        with gzip.open(backup_file, "rt", encoding="utf-8") as f:
            backup_data = json.load(f)

        logger.info(f"   Backup contains {len(backup_data['points'])} points")

        if self.dry_run:
            logger.info("   [DRY RUN] Would restore to Qdrant")
            return {
                "points": len(backup_data['points']),
                "dry_run": True
            }

        # Connect to Qdrant
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        # Check if collection exists
        try:
            collection_info = client.get_collection(QDRANT_COLLECTION)
            existing_points = collection_info.points_count
            
            if self.merge:
                logger.info(f"📥 Import mode: Collection '{QDRANT_COLLECTION}' contains {existing_points} existing points - will merge with backup")
            else:
                logger.warning(f"⚠️  Collection '{QDRANT_COLLECTION}' contains {existing_points} existing points!")
                if not self.force:
                    response = input("   Delete existing points and restore? [y/N]: ")
                    if response.lower() != 'y':
                        logger.info("   Restore cancelled")
                        return {"cancelled": True}
                else:
                    logger.info("   --force flag set, proceeding with restore")

                # Clear existing points
                logger.info("   🗑️  Clearing existing collection...")
                client.delete_collection(QDRANT_COLLECTION)

                # Recreate collection
                from qdrant_client.models import Distance, VectorParams
                client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(
                        size=backup_data['stats']['vector_size'],
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            logger.info(f"   Creating new collection (previous: {e})")
            from qdrant_client.models import Distance, VectorParams
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=backup_data['stats']['vector_size'],
                    distance=Distance.COSINE
                )
            )

        # Restore points in batches
        logger.info(f"   📥 Restoring {len(backup_data['points'])} points...")
        batch_size = 100

        for i in range(0, len(backup_data['points']), batch_size):
            batch = backup_data['points'][i:i+batch_size]
            logger.info(f"      Progress: {i}/{len(backup_data['points'])}")

            points = [
                PointStruct(
                    id=point['id'],
                    vector=point['vector'],
                    payload=point['payload']
                )
                for point in batch
            ]

            client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points
            )

        logger.info(f"   ✅ Restored {len(backup_data['points'])} points (upserted - existing points updated)")

        return {
            "points": len(backup_data['points']),
            "dry_run": False,
            "merge_mode": self.merge
        }

    def run_restore(self, timestamp: str = None, falkordb_only: bool = False,
                   qdrant_only: bool = False) -> Dict[str, Any]:
        """Run full restore process."""
        logger.info(f"🚀 Starting AutoMem restore")

        results = {
            "falkordb": None,
            "qdrant": None
        }

        try:
            # Restore FalkorDB
            if not qdrant_only:
                if timestamp:
                    falkor_backup = self.find_backup_by_timestamp("falkordb", timestamp)
                else:
                    falkor_backup = self.find_latest_backup("falkordb")

                results["falkordb"] = self.restore_falkordb(falkor_backup)

            # Restore Qdrant
            if not falkordb_only:
                if timestamp:
                    qdrant_backup = self.find_backup_by_timestamp("qdrant", timestamp)
                else:
                    qdrant_backup = self.find_latest_backup("qdrant")

                results["qdrant"] = self.restore_qdrant(qdrant_backup)

            logger.info("✅ Restore completed successfully")
            return results

        except Exception as e:
            logger.error(f"❌ Restore failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="AutoMem restore tool - restores FalkorDB and Qdrant from compressed JSON backups",
        epilog="""
Examples:
  # Restore from latest backup
  python scripts/restore_from_backup.py

  # Restore from specific timestamp
  python scripts/restore_from_backup.py --backup-timestamp 20251019_085625

  # Dry run (preview only)
  python scripts/restore_from_backup.py --dry-run

  # Restore only FalkorDB
  python scripts/restore_from_backup.py --falkordb-only

  # Restore only Qdrant
  python scripts/restore_from_backup.py --qdrant-only

  # Import/merge without deleting existing data
  python scripts/restore_from_backup.py --import
        """
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=str(BACKUP_DIR),
        help="Directory containing backup files (default: ./backups)"
    )
    parser.add_argument(
        "--backup-timestamp",
        type=str,
        help="Specific backup timestamp to restore (e.g., 20251019_085625)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview restore without making changes"
    )
    parser.add_argument(
        "--falkordb-only",
        action="store_true",
        help="Restore only FalkorDB (skip Qdrant)"
    )
    parser.add_argument(
        "--qdrant-only",
        action="store_true",
        help="Restore only Qdrant (skip FalkorDB)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force restore without confirmation prompts"
    )
    parser.add_argument(
        "--import",
        dest="merge",
        action="store_true",
        help="Import/merge backup data without deleting existing data (skips duplicates)"
    )

    args = parser.parse_args()

    restore = AutoMemRestore(
        backup_dir=Path(args.backup_dir),
        dry_run=args.dry_run,
        force=args.force,
        merge=args.merge
    )

    try:
        results = restore.run_restore(
            timestamp=args.backup_timestamp,
            falkordb_only=args.falkordb_only,
            qdrant_only=args.qdrant_only
        )
        print(json.dumps(results, indent=2))
        sys.exit(0)
    except Exception as e:
        logger.error(f"Restore failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
