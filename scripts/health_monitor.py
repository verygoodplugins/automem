#!/usr/bin/env python3
"""Health monitoring and auto-recovery service for AutoMem.

Runs as a background service that:
1. Monitors FalkorDB and Qdrant health
2. Verifies data consistency between graph and vector stores
3. Triggers recovery if inconsistencies detected
4. Sends alerts on failures

Usage:
    python scripts/health_monitor.py --interval 300  # Check every 5 minutes
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from falkordb import FalkorDB

# Load environment
load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout  # Write to stdout so Railway correctly parses log levels
)
logger = logging.getLogger("automem.health_monitor")


class HealthMonitor:
    """Monitors AutoMem system health and triggers recovery."""
    
    def __init__(self, auto_recover: bool = False, alert_webhook: Optional[str] = None):
        self.falkordb_host = os.getenv("FALKORDB_HOST", "localhost")
        self.falkordb_port = int(os.getenv("FALKORDB_PORT", "6379"))
        self.falkordb_password = os.getenv("FALKORDB_PASSWORD")
        self.graph_name = os.getenv("FALKORDB_GRAPH", "memories")
        
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION", "memories")
        
        self.api_url = os.getenv("AUTOMEM_API_URL") or os.getenv("MCP_MEMORY_HTTP_ENDPOINT", "http://localhost:8001")
        self.api_token = os.getenv("AUTOMEM_API_TOKEN")
        
        # Safety controls
        self.auto_recover = auto_recover  # Default: False (alert only)
        self.alert_webhook = alert_webhook or os.getenv("HEALTH_MONITOR_WEBHOOK")
        self.alert_email = os.getenv("HEALTH_MONITOR_EMAIL")
        self.drift_threshold_percent = float(os.getenv("HEALTH_MONITOR_DRIFT_THRESHOLD", "5"))
        self.critical_threshold_percent = float(os.getenv("HEALTH_MONITOR_CRITICAL_THRESHOLD", "50"))
        
        self.last_check: Dict[str, Any] = {}
        self.last_alert_time: Optional[datetime] = None
    
    def check_falkordb(self) -> Dict[str, Any]:
        """Check FalkorDB connection and memory count."""
        try:
            db = FalkorDB(
                host=self.falkordb_host,
                port=self.falkordb_port,
                password=self.falkordb_password,
                username="default" if self.falkordb_password else None
            )
            graph = db.select_graph(self.graph_name)
            
            # Count memories
            result = graph.query("MATCH (m:Memory) RETURN count(m) as count")
            memory_count = result.result_set[0][0] if result.result_set else 0
            
            return {
                "status": "healthy",
                "memory_count": memory_count,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"FalkorDB health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def check_qdrant(self) -> Dict[str, Any]:
        """Check Qdrant connection and memory count."""
        try:
            client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
            collection_info = client.get_collection(self.qdrant_collection)
            
            return {
                "status": "healthy",
                "points_count": collection_info.points_count,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def check_api(self) -> Dict[str, Any]:
        """Check AutoMem API health."""
        try:
            response = requests.get(
                f"{self.api_url}/health",
                timeout=10
            )
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "data": response.json(),
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"Status {response.status_code}",
                    "timestamp": datetime.utcnow().isoformat()
                }
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def check_consistency(self, falkordb_result: Dict, qdrant_result: Dict) -> Dict[str, Any]:
        """Check if FalkorDB and Qdrant memory counts are consistent."""
        if falkordb_result["status"] != "healthy" or qdrant_result["status"] != "healthy":
            return {
                "status": "skipped",
                "reason": "One or both stores unhealthy"
            }
        
        falkor_count = falkordb_result.get("memory_count", 0)
        qdrant_count = qdrant_result.get("points_count", 0)
        
        # Allow for some drift (in-flight writes, deletions, etc)
        drift = abs(falkor_count - qdrant_count)
        drift_percent = (drift / max(qdrant_count, 1)) * 100
        
        # Determine severity
        if drift_percent > self.critical_threshold_percent:
            severity = "critical"
        elif drift_percent > self.drift_threshold_percent:
            severity = "warning"
        else:
            severity = "ok"
        
        if severity != "ok":
            logger.warning(
                f"Inconsistency detected: FalkorDB={falkor_count}, "
                f"Qdrant={qdrant_count}, drift={drift_percent:.1f}%"
            )
            return {
                "status": "inconsistent",
                "severity": severity,
                "falkordb_count": falkor_count,
                "qdrant_count": qdrant_count,
                "drift": drift,
                "drift_percent": drift_percent
            }
        
        return {
            "status": "consistent",
            "falkordb_count": falkor_count,
            "qdrant_count": qdrant_count,
            "drift": drift,
            "drift_percent": drift_percent
        }
    
    def send_alert(self, level: str, title: str, message: str, details: Dict[str, Any] = None):
        """Send alert via configured channels."""
        alert_data = {
            "level": level,  # "warning", "critical", "info"
            "title": title,
            "message": message,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat(),
            "system": "AutoMem Health Monitor"
        }
        
        # Log alert
        if level == "critical":
            logger.error(f"🚨 CRITICAL: {title} - {message}")
        elif level == "warning":
            logger.warning(f"⚠️  WARNING: {title} - {message}")
        else:
            logger.info(f"ℹ️  INFO: {title} - {message}")
        
        # Send webhook
        if self.alert_webhook:
            try:
                requests.post(
                    self.alert_webhook,
                    json=alert_data,
                    timeout=10
                )
                logger.info(f"   📤 Alert sent to webhook")
            except Exception as e:
                logger.error(f"   ❌ Webhook failed: {e}")
        
        # Send email (if configured - would need email library)
        if self.alert_email:
            logger.info(f"   📧 Email alert to {self.alert_email} (email support not implemented)")
            # TODO: Implement email alerts with smtplib
        
        # Rate limiting - don't spam alerts
        self.last_alert_time = datetime.utcnow()
    
    def trigger_recovery(self, issue: str, drift_percent: float):
        """Trigger recovery process (manual or automatic)."""
        if not self.auto_recover:
            # Alert only mode - don't actually recover
            self.send_alert(
                level="critical",
                title="Data Loss Detected - Manual Recovery Required",
                message=f"{issue}. Drift: {drift_percent:.1f}%",
                details={
                    "drift_percent": drift_percent,
                    "auto_recover_enabled": False,
                    "recovery_command": "python scripts/recover_from_qdrant.py"
                }
            )
            logger.warning("🚨 AUTO-RECOVERY DISABLED - Please run recovery manually:")
            logger.warning("   python scripts/recover_from_qdrant.py")
            return False
        
        # Auto-recovery enabled - send alert and recover
        self.send_alert(
            level="critical",
            title="Auto-Recovery Triggered",
            message=f"{issue}. Starting automatic recovery...",
            details={
                "drift_percent": drift_percent,
                "auto_recover_enabled": True
            }
        )
        
        logger.warning(f"🔧 AUTO-RECOVERY ENABLED: Starting recovery for: {issue}")
        
        # Run recovery script
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "scripts/recover_from_qdrant.py"],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Recovery completed successfully")
                self.send_alert(
                    level="info",
                    title="Auto-Recovery Completed",
                    message="All memories restored successfully",
                    details={"stdout": result.stdout[-500:] if result.stdout else ""}
                )
                return True
            else:
                logger.error(f"❌ Recovery failed: {result.stderr}")
                self.send_alert(
                    level="critical",
                    title="Auto-Recovery Failed",
                    message="Recovery script failed - manual intervention required",
                    details={"stderr": result.stderr[-500:] if result.stderr else ""}
                )
                return False
        except Exception as e:
            logger.error(f"❌ Recovery execution failed: {e}")
            self.send_alert(
                level="critical",
                title="Auto-Recovery Error",
                message=f"Failed to execute recovery: {str(e)}",
                details={"error": str(e)}
            )
            return False
    
    def run_check(self):
        """Run full health check."""
        logger.info("🔍 Running health check...")
        
        falkor_health = self.check_falkordb()
        qdrant_health = self.check_qdrant()
        api_health = self.check_api()
        consistency = self.check_consistency(falkor_health, qdrant_health)
        
        # Log results
        logger.info(f"   FalkorDB: {falkor_health['status']}")
        logger.info(f"   Qdrant:   {qdrant_health['status']}")
        logger.info(f"   API:      {api_health['status']}")
        logger.info(f"   Consistency: {consistency['status']}")
        
        # Store results
        self.last_check = {
            "timestamp": datetime.utcnow().isoformat(),
            "falkordb": falkor_health,
            "qdrant": qdrant_health,
            "api": api_health,
            "consistency": consistency
        }
        
        # Check if recovery needed
        if consistency["status"] == "inconsistent":
            severity = consistency.get("severity", "warning")
            drift_percent = consistency.get("drift_percent", 0)
            
            if severity == "critical":
                # Critical data loss detected
                logger.warning(
                    f"⚠️  CRITICAL: FalkorDB has {drift_percent:.1f}% drift from Qdrant"
                )
                self.trigger_recovery("Major data loss detected", drift_percent)
            elif severity == "warning":
                # Minor drift - alert but don't recover
                self.send_alert(
                    level="warning",
                    title="Memory Count Drift Detected",
                    message=f"FalkorDB and Qdrant are {drift_percent:.1f}% out of sync",
                    details={
                        "falkordb_count": consistency["falkordb_count"],
                        "qdrant_count": consistency["qdrant_count"],
                        "drift": consistency["drift"],
                        "drift_percent": drift_percent
                    }
                )
        
        return self.last_check
    
    def run_forever(self, interval: int = 300):
        """Run health checks continuously."""
        logger.info(f"🚀 Starting health monitor (interval: {interval}s)")
        logger.info(f"   Auto-recovery: {'ENABLED' if self.auto_recover else 'DISABLED (alert only)'}")
        logger.info(f"   Drift threshold: {self.drift_threshold_percent}%")
        logger.info(f"   Critical threshold: {self.critical_threshold_percent}%")
        if self.alert_webhook:
            logger.info(f"   Webhook alerts: {self.alert_webhook}")
        
        while True:
            try:
                self.run_check()
            except Exception as e:
                logger.error(f"Health check failed: {e}")
            
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(
        description="AutoMem health monitor - monitors system health and optionally triggers recovery",
        epilog="""
Examples:
  # Alert only (safe default)
  python scripts/health_monitor.py --interval 300

  # Enable auto-recovery (use with caution!)
  python scripts/health_monitor.py --auto-recover --interval 300

  # One-time check
  python scripts/health_monitor.py --once

  # With webhook alerts
  python scripts/health_monitor.py --webhook https://hooks.slack.com/...
        """
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds (default: 300)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit"
    )
    parser.add_argument(
        "--auto-recover",
        action="store_true",
        help="Enable automatic recovery (default: alert only). USE WITH CAUTION!"
    )
    parser.add_argument(
        "--webhook",
        type=str,
        help="Webhook URL for alerts (e.g., Slack webhook)"
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=5.0,
        help="Warning drift threshold percentage (default: 5.0)"
    )
    parser.add_argument(
        "--critical-threshold",
        type=float,
        default=50.0,
        help="Critical drift threshold percentage for recovery (default: 50.0)"
    )
    
    args = parser.parse_args()
    
    # Set thresholds via env if provided via args
    if args.drift_threshold:
        os.environ["HEALTH_MONITOR_DRIFT_THRESHOLD"] = str(args.drift_threshold)
    if args.critical_threshold:
        os.environ["HEALTH_MONITOR_CRITICAL_THRESHOLD"] = str(args.critical_threshold)
    
    monitor = HealthMonitor(
        auto_recover=args.auto_recover,
        alert_webhook=args.webhook
    )
    
    if args.once:
        result = monitor.run_check()
        import json
        print(json.dumps(result, indent=2))
    else:
        if args.auto_recover:
            logger.warning("⚠️  AUTO-RECOVERY ENABLED - System will automatically recover from data loss")
            logger.warning("   Press Ctrl+C within 10 seconds to cancel...")
            try:
                time.sleep(10)
            except KeyboardInterrupt:
                logger.info("Cancelled by user")
                sys.exit(0)
        
        monitor.run_forever(interval=args.interval)


if __name__ == "__main__":
    main()
