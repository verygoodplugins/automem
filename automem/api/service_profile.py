from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict

from flask import Blueprint, jsonify


def create_service_blueprint(
    *,
    get_memory_graph: Callable[[], Any],
    get_qdrant_client: Callable[[], Any],
    graph_name: str,
    collection_name: str,
    embedding_model: str,
    utc_now: Callable[[], str],
    get_service_profile: Callable[[], Dict[str, Any]],
    get_service_mode: Callable[[], str],
    get_service_tier: Callable[[], str],
    get_embedding_provider_name: Callable[[], str | None],
) -> Blueprint:
    bp = Blueprint("service", __name__)

    @bp.route("/service/profile", methods=["GET"])
    def service_profile() -> Any:
        graph_available = get_memory_graph() is not None
        qdrant_available = get_qdrant_client() is not None
        profile = deepcopy(get_service_profile())
        capabilities = deepcopy(profile.get("capabilities", {}))
        profile["capabilities"] = capabilities

        return jsonify(
            {
                "status": "success",
                "service": {
                    "tier": get_service_tier(),
                    "mode": get_service_mode(),
                    "profile": profile,
                    "graph": {
                        "name": graph_name,
                        "available": graph_available,
                    },
                    "vector_store": {
                        "collection": collection_name,
                        "available": qdrant_available,
                        "expected": bool(capabilities.get("qdrant_expected", True)),
                    },
                    "embedding": {
                        "provider": get_embedding_provider_name(),
                        "model": embedding_model,
                        "tier": profile.get("embedding_tier"),
                    },
                    "consolidation": {
                        "tier": profile.get("consolidation_tier"),
                    },
                    "timestamp": utc_now(),
                },
            }
        )

    return bp
