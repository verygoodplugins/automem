"""Multi-Tenant AutoMem Service.

Wraps the core AutoMem app with tenant isolation and admin endpoints.
Each tenant gets their own isolated graph + vector storage.
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Any, Dict

from dotenv import load_dotenv
from flask import Flask, abort, g, jsonify, request
from falkordb import FalkorDB
from qdrant_client import QdrantClient
from werkzeug.exceptions import HTTPException

from tenant_manager import TenantManager, require_admin, require_tenant

# Load environment first
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("automem.multitenant")

app = Flask(__name__)

# Initialize tenant manager
FALKORDB_HOST = (
    os.getenv("FALKORDB_HOST") or
    os.getenv("RAILWAY_PRIVATE_DOMAIN") or
    "localhost"
)
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize connections
falkordb = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if QDRANT_URL else None
tenant_manager = TenantManager(falkordb, qdrant)


@app.before_request
def authenticate_tenant():
    """Authenticate tenant for all non-admin requests."""
    # Skip auth for health check and admin endpoints
    if request.endpoint in {'health', 'admin_create_tenant', 'admin_list_tenants', 
                           'admin_get_tenant_stats', 'admin_delete_tenant'}:
        return
    
    # Authenticate and store tenant in request context
    tenant = tenant_manager.authenticate_request()
    if tenant:
        g.tenant = tenant
        logger.info(f"Request authenticated for tenant: {tenant.tenant_id}")
    else:
        g.tenant = None


@app.errorhandler(HTTPException)
def handle_http_error(error):
    """Return JSON for HTTP errors."""
    return jsonify({"error": error.description}), error.code


@app.errorhandler(Exception)
def handle_generic_error(error):
    """Catch-all for unexpected errors."""
    logger.exception("Unexpected error")
    return jsonify({"error": "Internal server error"}), 500


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "automem-multitenant",
        "tenants_enabled": True,
    })


# =============================================================================
# ADMIN ENDPOINTS (Tenant Management)
# =============================================================================

@app.route("/admin/tenants", methods=["POST"])
def admin_create_tenant():
    """Create a new tenant.
    
    POST /admin/tenants
    Authorization: Bearer {ADMIN_API_TOKEN}
    
    {
      "tenant_id": "fernando",
      "name": "Fernando's Memory Service",
      "api_token": "optional-custom-token",
      "metadata": {"plan": "pro", "email": "fernando@example.com"}
    }
    """
    require_admin()
    
    data = request.get_json() or {}
    tenant_id = data.get("tenant_id")
    name = data.get("name") or tenant_id
    api_token = data.get("api_token") or f"amt_{secrets.token_urlsafe(32)}"
    metadata = data.get("metadata") or {}
    
    if not tenant_id:
        abort(400, description="tenant_id required")
    
    try:
        tenant = tenant_manager.create_tenant(
            tenant_id=tenant_id,
            name=name,
            api_token=api_token,
            metadata=metadata,
        )
        
        return jsonify({
            "success": True,
            "tenant": tenant.to_dict(),
            "api_token": api_token,  # Return token for customer setup
        }), 201
    
    except ValueError as e:
        abort(400, description=str(e))
    except Exception as e:
        logger.exception(f"Failed to create tenant {tenant_id}")
        abort(500, description=f"Failed to create tenant: {e}")


@app.route("/admin/tenants", methods=["GET"])
def admin_list_tenants():
    """List all tenants with stats.
    
    GET /admin/tenants
    Authorization: Bearer {ADMIN_API_TOKEN}
    """
    require_admin()
    
    tenants = tenant_manager.list_tenants()
    
    # Optionally include stats
    include_stats = request.args.get("include_stats", "false").lower() == "true"
    
    result = []
    for tenant in tenants:
        tenant_data = tenant.to_dict()
        if include_stats:
            try:
                stats = tenant_manager.get_tenant_stats(tenant.tenant_id)
                tenant_data.update(stats)
            except Exception:
                logger.exception(f"Failed to get stats for {tenant.tenant_id}")
        result.append(tenant_data)
    
    return jsonify({"tenants": result})


@app.route("/admin/tenants/<tenant_id>/stats", methods=["GET"])
def admin_get_tenant_stats(tenant_id: str):
    """Get detailed stats for a tenant.
    
    GET /admin/tenants/{tenant_id}/stats
    Authorization: Bearer {ADMIN_API_TOKEN}
    """
    require_admin()
    
    try:
        stats = tenant_manager.get_tenant_stats(tenant_id)
        return jsonify(stats)
    except ValueError as e:
        abort(404, description=str(e))
    except Exception as e:
        logger.exception(f"Failed to get stats for {tenant_id}")
        abort(500, description=str(e))


@app.route("/admin/tenants/<tenant_id>", methods=["DELETE"])
def admin_delete_tenant(tenant_id: str):
    """Delete a tenant and all their data (DANGEROUS).
    
    DELETE /admin/tenants/{tenant_id}?confirm=true
    Authorization: Bearer {ADMIN_API_TOKEN}
    """
    require_admin()
    
    confirm = request.args.get("confirm", "").lower() == "true"
    if not confirm:
        abort(400, description="Must confirm deletion with ?confirm=true")
    
    try:
        tenant_manager.delete_tenant(tenant_id, confirm=True)
        return jsonify({"success": True, "message": f"Tenant {tenant_id} deleted"})
    except ValueError as e:
        abort(404, description=str(e))
    except Exception as e:
        logger.exception(f"Failed to delete tenant {tenant_id}")
        abort(500, description=str(e))


# =============================================================================
# TENANT ENDPOINTS (Memory Operations)
# =============================================================================

# Import the core AutoMem functionality
# We'll need to modify core app.py to support multi-tenancy context
# For now, create placeholder endpoints that demonstrate the pattern

@app.route("/memory", methods=["POST"])
def store_memory():
    """Store a memory for the authenticated tenant.
    
    POST /memory
    Authorization: Bearer {tenant_api_token}
    
    {
      "content": "Memory content",
      "tags": ["decision", "database"],
      "importance": 0.9
    }
    """
    tenant = require_tenant()
    
    # TODO: Import and call actual store_memory from core app.py
    # For now, return success placeholder
    data = request.get_json() or {}
    
    logger.info(f"Storing memory for tenant {tenant.tenant_id}")
    
    # In real implementation, this would:
    # 1. Get tenant's graph: tenant_graph = falkordb.select_graph(tenant.graph_name())
    # 2. Get tenant's collection: collection = tenant.collection_name()
    # 3. Store memory with tenant isolation
    
    return jsonify({
        "success": True,
        "tenant_id": tenant.tenant_id,
        "message": "Memory stored (placeholder)",
        "id": secrets.token_hex(16),
    }), 201


@app.route("/recall", methods=["GET"])
def recall_memories():
    """Recall memories for the authenticated tenant.
    
    GET /recall?query=project&tags=decision
    Authorization: Bearer {tenant_api_token}
    """
    tenant = require_tenant()
    
    query = request.args.get("query", "")
    logger.info(f"Recalling memories for tenant {tenant.tenant_id}: {query}")
    
    # TODO: Import and call actual recall from core app.py with tenant context
    
    return jsonify({
        "success": True,
        "tenant_id": tenant.tenant_id,
        "results": [],
        "message": "Recall placeholder - integrate core app.py next",
    })


@app.route("/memory/<memory_id>", methods=["GET", "PUT", "DELETE"])
def manage_memory(memory_id: str):
    """Get, update, or delete a specific memory (tenant-scoped).
    
    GET/PUT/DELETE /memory/{memory_id}
    Authorization: Bearer {tenant_api_token}
    """
    tenant = require_tenant()
    
    if request.method == "GET":
        logger.info(f"Getting memory {memory_id} for tenant {tenant.tenant_id}")
        # TODO: Implement with tenant isolation
        return jsonify({"success": True, "message": "Placeholder"})
    
    elif request.method == "PUT":
        logger.info(f"Updating memory {memory_id} for tenant {tenant.tenant_id}")
        # TODO: Implement with tenant isolation
        return jsonify({"success": True, "message": "Placeholder"})
    
    elif request.method == "DELETE":
        logger.info(f"Deleting memory {memory_id} for tenant {tenant.tenant_id}")
        # TODO: Implement with tenant isolation
        return jsonify({"success": True, "message": "Placeholder"})


@app.route("/associate", methods=["POST"])
def associate_memories():
    """Create relationship between memories (tenant-scoped).
    
    POST /associate
    Authorization: Bearer {tenant_api_token}
    
    {
      "memory1_id": "uuid1",
      "memory2_id": "uuid2",
      "type": "PREFERS_OVER",
      "strength": 0.9
    }
    """
    tenant = require_tenant()
    
    logger.info(f"Associating memories for tenant {tenant.tenant_id}")
    # TODO: Implement with tenant isolation
    
    return jsonify({"success": True, "message": "Placeholder"})


# =============================================================================
# STARTUP
# =============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    
    logger.info("=" * 60)
    logger.info("AutoMem Multi-Tenant Service Starting")
    logger.info("=" * 60)
    logger.info(f"FalkorDB: {FALKORDB_HOST}:{FALKORDB_PORT}")
    logger.info(f"Qdrant: {QDRANT_URL or 'Not configured'}")
    logger.info(f"Admin API: {'Enabled' if os.getenv('ADMIN_API_TOKEN') else 'DISABLED'}")
    logger.info("=" * 60)
    
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG") == "1")
