from __future__ import annotations

from typing import Any, Callable, Optional


def extract_api_token(request_obj: Any, configured_api_token: Optional[str]) -> Optional[str]:
    if not configured_api_token:
        return None

    auth_header = request_obj.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()

    api_key_header = request_obj.headers.get("X-API-Key")
    if api_key_header:
        return api_key_header.strip()

    api_key_param = request_obj.args.get("api_key")
    if api_key_param:
        return api_key_param.strip()

    return None


def require_admin_token(
    *,
    request_obj: Any,
    admin_token: Optional[str],
    abort_fn: Callable[..., Any],
) -> None:
    if not admin_token:
        abort_fn(403, description="Admin token not configured")

    provided = (
        request_obj.headers.get("X-Admin-Token")
        or request_obj.headers.get("X-Admin-Api-Key")
        or request_obj.args.get("admin_token")
    )

    if provided != admin_token:
        abort_fn(401, description="Admin authorization required")


def require_api_token(
    *,
    request_obj: Any,
    api_token: Optional[str],
    extract_api_token_fn: Callable[[], Optional[str]],
    abort_fn: Callable[..., Any],
) -> None:
    if not api_token:
        return

    endpoint = request_obj.endpoint or ""
    if endpoint.endswith("health") or request_obj.path == "/health":
        return

    token = extract_api_token_fn()
    if token != api_token:
        abort_fn(401, description="Unauthorized")
