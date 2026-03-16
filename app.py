"""Spyhop FastAPI server, deployed as a Modal app.

The OMol25 index parquet is stored in a Modal Volume ('spyhop-index') and
mounted read-only at /index.  The Executor is initialized once per container
in the @modal.enter() hook and reused across all requests.

Deploy:
    modal deploy spyhop/app.py

Local dev:
    modal serve spyhop/app.py
"""

from __future__ import annotations

import os
import dataclasses

import modal
from pydantic import BaseModel

# ── Modal image & volume ───────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]", "duckdb>=1.1", "pyarrow>=19.0")
    .add_local_python_source("spyhop")
)

MODAL_ENV = os.environ.get('MODAL_ENV', 'dev')
volume = modal.Volume.from_name("spyhop-index", environment_name=MODAL_ENV)

INDEX_PATH = "/index/index.parquet"

app = modal.App("spyhop")

# ── Request / response models ──────────────────────────────────────────
# Pydantic models are an HTTP-layer concern (FastAPI validation + OpenAPI
# schema).  Domain types (FilterSpec, CountResult, ManifestRow) live in
# spyhop.query as plain dataclasses and are the shared client/server contract.


class RangeFilterModel(BaseModel):
    min: int | float | None = None
    max: int | float | None = None


class FilterSpecModel(BaseModel):
    must_have: list[str] = []
    must_not_have: list[str] = []
    domain: str | None = None
    num_atoms: RangeFilterModel | None = None
    charge: int | None = None
    spin: int | None = None


class CountResponse(BaseModel):
    count: int
    estimated_gb: float


class ManifestRowModel(BaseModel):
    eagle_path: str
    size_orca_tar_zst: int
    size_gbw: int
    size_density_mat_npz: int


class ManifestResponse(BaseModel):
    count: int
    manifest: list[ManifestRowModel]


# ── DuckDB executor ────────────────────────────────────────────────────
# Lives here rather than in spyhop.query because duckdb is a server-only
# dependency — the CLI and client never run queries locally.


def _build_where(spec) -> tuple[str, list]:
    """Return (WHERE clause string, list of positional params).

    Column names are only ever interpolated from VALID_ELEMENTS (controlled
    by us) or are literal SQL keywords/constants.  The one user-supplied
    string value (domain) is passed as a ? parameter to avoid any injection
    risk even though it is already validated against VALID_DOMAINS.
    """
    from spyhop.query import FilterSpec  # noqa: F401 — for type hint only
    clauses: list[str] = []
    params: list = []

    for sym in spec.must_have:
        clauses.append(f"has_{sym} = true")
    for sym in spec.must_not_have:
        clauses.append(f"has_{sym} = false")
    if spec.domain is not None:
        clauses.append("domain = ?")
        params.append(spec.domain)
    if spec.num_atoms is not None:
        if spec.num_atoms.min is not None:
            clauses.append(f"num_atoms >= {int(spec.num_atoms.min)}")
        if spec.num_atoms.max is not None:
            clauses.append(f"num_atoms <= {int(spec.num_atoms.max)}")
    if spec.charge is not None:
        clauses.append(f"charge = {int(spec.charge)}")
    if spec.spin is not None:
        clauses.append(f"spin = {int(spec.spin)}")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


class Executor:
    """DuckDB-backed query executor over a parquet index file."""

    def __init__(self, parquet_path: str) -> None:
        import duckdb
        self._con = duckdb.connect(":memory:")
        self._con.execute(
            f"CREATE VIEW index_view AS SELECT * FROM read_parquet('{parquet_path}')"
        )

    def count(self, spec) -> "CountResult":
        from spyhop.query import CountResult
        where, params = _build_where(spec)
        row = self._con.execute(
            f"""
            SELECT
                COUNT(*) AS n,
                COALESCE(SUM(size_orca_tar_zst + size_gbw + size_density_mat_npz), 0) AS total_bytes
            FROM index_view
            {where}
            """,
            params,
        ).fetchone()
        return CountResult(count=row[0], estimated_bytes=row[1])

    def manifest(self, spec) -> list:
        from spyhop.query import ManifestRow
        where, params = _build_where(spec)
        rows = self._con.execute(
            f"""
            SELECT eagle_path, size_orca_tar_zst, size_gbw, size_density_mat_npz
            FROM index_view
            {where}
            """,
            params,
        ).fetchall()
        return [ManifestRow(*r) for r in rows]


# ── Modal app class ────────────────────────────────────────────────────


@app.cls(image=image, volumes={"/index": volume}, min_containers=1)
@modal.concurrent(max_inputs=50)
class SpyhopServer:
    @modal.enter()
    def startup(self) -> None:
        self.executor = Executor(INDEX_PATH)

    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from spyhop.query import FilterSpec, RangeFilter

        executor = self.executor

        web_app = FastAPI(title="Spyhop")
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["POST"],
            allow_headers=["*"],
        )

        def to_filter_spec(model: FilterSpecModel) -> FilterSpec:
            try:
                return FilterSpec(
                    must_have=model.must_have,
                    must_not_have=model.must_not_have,
                    domain=model.domain,
                    num_atoms=RangeFilter(min=model.num_atoms.min, max=model.num_atoms.max)
                    if model.num_atoms else None,
                    charge=model.charge,
                    spin=model.spin,
                )
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))

        @web_app.post("/query/count", response_model=CountResponse)
        def query_count(body: FilterSpecModel):
            spec = to_filter_spec(body)
            result = executor.count(spec)
            return CountResponse(count=result.count, estimated_gb=round(result.estimated_gb, 2))

        @web_app.post("/query/manifest", response_model=ManifestResponse)
        def query_manifest(body: FilterSpecModel):
            spec = to_filter_spec(body)
            rows = executor.manifest(spec)
            return ManifestResponse(
                count=len(rows),
                manifest=[ManifestRowModel(**dataclasses.asdict(r)) for r in rows],
            )

        return web_app
