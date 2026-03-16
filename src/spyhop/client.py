"""HTTP client for the Spyhop API.

Thin wrapper around the two REST endpoints.  Uses the same domain types
(FilterSpec, CountResult, ManifestRow) as the server — no Pydantic required.
"""

from __future__ import annotations

import dataclasses
import os

import httpx

from spyhop.query import CountResult, FilterSpec, ManifestRow

DEFAULT_BASE_URL = os.environ.get("SPYHOP_URL", "https://garden-ai--spyhop-spyhopserver-serve-dev.modal.run")


def count(spec: FilterSpec, base_url: str = DEFAULT_BASE_URL) -> CountResult:
    """Return the count and estimated transfer size for the given filter."""
    resp = httpx.post(
        f"{base_url.rstrip('/')}/query/count",
        json=dataclasses.asdict(spec),
    )
    resp.raise_for_status()
    data = resp.json()
    return CountResult(count=data["count"], estimated_bytes=int(data["estimated_gb"] * 1e9))


def manifest(spec: FilterSpec, base_url: str = DEFAULT_BASE_URL) -> list[ManifestRow]:
    """Return the full manifest of matching structures."""
    resp = httpx.post(
        f"{base_url.rstrip('/')}/query/manifest",
        json=dataclasses.asdict(spec),
    )
    resp.raise_for_status()
    return [ManifestRow(**row) for row in resp.json()["manifest"]]
