"""Multi-Tenant Management for AutoMem.

Provides tenant isolation with separate graph/vector spaces per customer.
Each tenant gets:
- Isolated FalkorDB graph database
- Isolated Qdrant collection
- Unique API token for authentication
- Independent consolidation/enrichment pipelines
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from falkordb import FalkorDB
from flask import abort, g, request
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

logger = logging.getLogger("automem.tenants")

# Admin database for tenant metadata
ADMIN_GRAPH_NAME = "automem_admin"
ADMIN_API_TOKEN = os.getenv("ADMIN_API_TOKEN")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "768"))


@dataclass
class Tenant:
    """Tenant metadata and credentials."""
    
    tenant_id: str
    name: str
    api_token: str
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "created_at": self.created_at,
            "active": self.active,
            "metadata": self.metadata,
        }
    
    def graph_name(self) -> str:
        """Get tenant's isolated graph database name."""
        return f"memories_{self.tenant_id}"
    
    def collection_name(self) -> str:
        """Get tenant's isolated Qdrant collection name."""
        return f"memories_{self.tenant_id}"


class TenantManager:
    """Manages tenant lifecycle and authentication."""
    
    def __init__(self, falkordb: FalkorDB, qdrant: Optional[QdrantClient] = None):
        self.falkordb = falkordb
        self.qdrant = qdrant
        self.admin_graph = falkordb.select_graph(ADMIN_GRAPH_NAME)
        self._ensure_admin_schema()
    
    def _ensure_admin_schema(self) -> None:
        """Create tenant storage schema in admin graph."""
        try:
            # Create index on api_token for fast lookups
            self.admin_graph.query("""
                CREATE INDEX FOR (t:Tenant) ON (t.api_token)
            """)
            logger.info("Admin schema initialized")
        except Exception as e:
            # Index might already exist
            logger.debug(f"Admin schema setup: {e}")
    
    def create_tenant(
        self,
        tenant_id: str,
        name: str,
        api_token: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tenant:
        """Create a new tenant with isolated resources."""
        # Validate tenant_id
        if not tenant_id or not tenant_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("tenant_id must be alphanumeric with _ or - only")
        
        if not api_token or len(api_token) < 16:
            raise ValueError("api_token must be at least 16 characters")
        
        # Check if tenant already exists
        existing = self._get_tenant_by_id(tenant_id)
        if existing:
            raise ValueError(f"Tenant {tenant_id} already exists")
        
        # Check if token already used
        existing_token = self._get_tenant_by_token(api_token)
        if existing_token:
            raise ValueError("API token already in use")
        
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            api_token=api_token,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )
        
        # Store tenant metadata
        try:
            self.admin_graph.query("""
                CREATE (t:Tenant {
                    tenant_id: $tenant_id,
                    name: $name,
                    api_token: $api_token,
                    created_at: $created_at,
                    active: $active,
                    metadata: $metadata
                })
            """, {
                "tenant_id": tenant.tenant_id,
                "name": tenant.name,
                "api_token": tenant.api_token,
                "created_at": tenant.created_at,
                "active": tenant.active,
                "metadata": str(tenant.metadata),
            })
        except Exception as e:
            logger.exception(f"Failed to create tenant {tenant_id}")
            raise ValueError(f"Failed to create tenant: {e}")
        
        # Initialize tenant's graph database
        try:
            tenant_graph = self.falkordb.select_graph(tenant.graph_name())
            # Create a placeholder node to initialize the graph
            tenant_graph.query("CREATE (:InitMarker {created_at: $ts})", {
                "ts": tenant.created_at
            })
            logger.info(f"Created graph database for tenant {tenant_id}")
        except Exception as e:
            logger.exception(f"Failed to create graph for tenant {tenant_id}")
            # Cleanup admin entry
            self.admin_graph.query("MATCH (t:Tenant {tenant_id: $id}) DELETE t", 
                                  {"id": tenant_id})
            raise ValueError(f"Failed to initialize tenant graph: {e}")
        
        # Initialize tenant's Qdrant collection
        if self.qdrant:
            try:
                self.qdrant.create_collection(
                    collection_name=tenant.collection_name(),
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection for tenant {tenant_id}")
            except Exception as e:
                logger.warning(f"Failed to create Qdrant collection for {tenant_id}: {e}")
                # Not fatal - tenant can work without vector search
        
        logger.info(f"Tenant {tenant_id} created successfully")
        return tenant
    
    def _get_tenant_by_id(self, tenant_id: str) -> Optional[Tenant]:
        """Fetch tenant by ID."""
        try:
            result = self.admin_graph.query("""
                MATCH (t:Tenant {tenant_id: $id})
                RETURN t
            """, {"id": tenant_id})
            
            if not result.result_set:
                return None
            
            node = result.result_set[0][0]
            return self._node_to_tenant(node)
        except Exception:
            logger.exception(f"Failed to fetch tenant {tenant_id}")
            return None
    
    def _get_tenant_by_token(self, api_token: str) -> Optional[Tenant]:
        """Fetch tenant by API token."""
        try:
            result = self.admin_graph.query("""
                MATCH (t:Tenant {api_token: $token})
                RETURN t
            """, {"token": api_token})
            
            if not result.result_set:
                return None
            
            node = result.result_set[0][0]
            return self._node_to_tenant(node)
        except Exception:
            logger.exception("Failed to fetch tenant by token")
            return None
    
    def _node_to_tenant(self, node: Any) -> Tenant:
        """Convert graph node to Tenant object."""
        props = node.properties if hasattr(node, 'properties') else node
        
        # Parse metadata if stored as string
        metadata = props.get("metadata", {})
        if isinstance(metadata, str):
            try:
                import json
                metadata = json.loads(metadata)
            except:
                metadata = {}
        
        return Tenant(
            tenant_id=props.get("tenant_id", ""),
            name=props.get("name", ""),
            api_token=props.get("api_token", ""),
            created_at=props.get("created_at", ""),
            metadata=metadata,
            active=props.get("active", True),
        )
    
    def list_tenants(self) -> List[Tenant]:
        """List all tenants."""
        try:
            result = self.admin_graph.query("""
                MATCH (t:Tenant)
                RETURN t
                ORDER BY t.created_at DESC
            """)
            
            return [
                self._node_to_tenant(row[0])
                for row in result.result_set
            ]
        except Exception:
            logger.exception("Failed to list tenants")
            return []
    
    def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get usage statistics for a tenant."""
        tenant = self._get_tenant_by_id(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        stats = {
            "tenant_id": tenant_id,
            "memory_count": 0,
            "graph_nodes": 0,
            "graph_relationships": 0,
            "vector_points": 0,
            "storage_mb": 0.0,
        }
        
        # Graph stats
        try:
            tenant_graph = self.falkordb.select_graph(tenant.graph_name())
            
            # Count memories
            result = tenant_graph.query("MATCH (m:Memory) RETURN count(m) as count")
            if result.result_set:
                stats["memory_count"] = result.result_set[0][0]
            
            # Count all nodes
            result = tenant_graph.query("MATCH (n) RETURN count(n) as count")
            if result.result_set:
                stats["graph_nodes"] = result.result_set[0][0]
            
            # Count relationships
            result = tenant_graph.query("MATCH ()-[r]->() RETURN count(r) as count")
            if result.result_set:
                stats["graph_relationships"] = result.result_set[0][0]
            
        except Exception:
            logger.exception(f"Failed to get graph stats for {tenant_id}")
        
        # Vector stats
        if self.qdrant:
            try:
                collection_info = self.qdrant.get_collection(tenant.collection_name())
                stats["vector_points"] = collection_info.points_count or 0
                
                # Rough storage estimate (vectors + payload)
                # 768-dim float32 = 3KB, + ~1KB payload = ~4KB per point
                stats["storage_mb"] = (stats["vector_points"] * 4) / 1024.0
            except Exception:
                logger.debug(f"Failed to get vector stats for {tenant_id}")
        
        return stats
    
    def delete_tenant(self, tenant_id: str, confirm: bool = False) -> None:
        """Delete tenant and all their data (DANGEROUS)."""
        if not confirm:
            raise ValueError("Must confirm tenant deletion")
        
        tenant = self._get_tenant_by_id(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        # Delete graph database
        try:
            tenant_graph = self.falkordb.select_graph(tenant.graph_name())
            tenant_graph.query("MATCH (n) DETACH DELETE n")
            logger.info(f"Deleted graph database for tenant {tenant_id}")
        except Exception:
            logger.exception(f"Failed to delete graph for tenant {tenant_id}")
        
        # Delete Qdrant collection
        if self.qdrant:
            try:
                self.qdrant.delete_collection(tenant.collection_name())
                logger.info(f"Deleted Qdrant collection for tenant {tenant_id}")
            except Exception:
                logger.exception(f"Failed to delete Qdrant collection for {tenant_id}")
        
        # Delete admin entry
        try:
            self.admin_graph.query("""
                MATCH (t:Tenant {tenant_id: $id})
                DELETE t
            """, {"id": tenant_id})
            logger.info(f"Deleted admin entry for tenant {tenant_id}")
        except Exception:
            logger.exception(f"Failed to delete admin entry for {tenant_id}")
    
    def authenticate_request(self) -> Optional[Tenant]:
        """Extract and validate tenant from request."""
        # Extract token from request
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            api_token = auth_header[7:].strip()
        else:
            api_token = (
                request.headers.get("X-API-Key") or
                request.args.get("api_key")
            )
        
        if not api_token:
            return None
        
        # Check if it's admin token
        if api_token == ADMIN_API_TOKEN:
            # Admin has access to all tenants, but doesn't map to a specific tenant
            return None
        
        # Look up tenant by token
        tenant = self._get_tenant_by_token(api_token)
        if not tenant or not tenant.active:
            abort(401, description="Invalid or inactive API token")
        
        return tenant


def require_tenant():
    """Flask decorator/helper to enforce tenant authentication."""
    if not hasattr(g, 'tenant') or g.tenant is None:
        abort(401, description="Tenant authentication required")
    return g.tenant


def require_admin():
    """Flask decorator/helper to enforce admin authentication."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
    else:
        token = request.headers.get("X-Admin-Token")
    
    if token != ADMIN_API_TOKEN:
        abort(403, description="Admin access required")
