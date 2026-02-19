from __future__ import annotations

import app

EXPECTED_ROUTE_METHODS = {
    ("GET", "/health"),
    ("POST", "/memory"),
    ("GET", "/memory/<memory_id>"),
    ("PATCH", "/memory/<memory_id>"),
    ("DELETE", "/memory/<memory_id>"),
    ("GET", "/memory/by-tag"),
    ("POST", "/associate"),
    ("GET", "/recall"),
    ("GET", "/startup-recall"),
    ("GET", "/analyze"),
    ("GET", "/memories/<memory_id>/related"),
    ("POST", "/admin/reembed"),
    ("POST", "/admin/sync"),
    ("POST", "/consolidate"),
    ("GET", "/consolidate/status"),
    ("GET", "/enrichment/status"),
    ("POST", "/enrichment/reprocess"),
    ("GET", "/graph/snapshot"),
    ("GET", "/graph/neighbors/<memory_id>"),
    ("GET", "/graph/stats"),
    ("GET", "/graph/types"),
    ("GET", "/graph/relations"),
    ("GET", "/stream"),
    ("GET", "/stream/status"),
}


def _actual_route_methods() -> set[tuple[str, str]]:
    route_methods: set[tuple[str, str]] = set()
    for rule in app.app.url_map.iter_rules():
        if rule.endpoint == "static":
            continue
        for method in sorted(
            method for method in rule.methods if method not in {"HEAD", "OPTIONS"}
        ):
            route_methods.add((method, rule.rule))
    return route_methods


def test_public_route_contract_is_frozen() -> None:
    actual = _actual_route_methods()

    missing = EXPECTED_ROUTE_METHODS - actual
    unexpected = actual - EXPECTED_ROUTE_METHODS

    assert not missing and not unexpected, (
        "Route contract mismatch. " f"Missing={sorted(missing)} " f"Unexpected={sorted(unexpected)}"
    )
