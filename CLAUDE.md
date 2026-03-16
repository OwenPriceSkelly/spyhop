# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project Overview

**spyhop** is a filter interface for the OMol25 DFT dataset (~4M structures, hosted at Argonne Eagle via Globus). It lets researchers filter by element composition, domain, and molecular properties, then download a Globus transfer manifest for their selection.

The companion repo at `../omol` contains exploration scripts, notes, and the full design history. Design docs are at `../omol/docs/plans/`, e.g.:
- `../omol/docs/plans/2026-03-09-omol-filter-interface-design.md` — overall architecture
- `../omol/docs/plans/2026-03-16-spyhop-frontend-design.md` — frontend design
- `../omol/docs/plans/2026-03-16-spyhop-frontend-implementation.md` — frontend implementation plan

New design docs should be written to the same location, NOT in this repository.

## Repo Structure

```
app.py          — Modal deployment (FastAPI + DuckDB, reads index from Modal Volume)
index.html      — Static frontend (Alpine.js + Tailwind CDN, no build step)
pyproject.toml  — Package definition; CLI entry point is spyhop:main
src/spyhop/
  __init__.py   — Typer CLI (spyhop count, spyhop manifest)
  query.py      — Domain types: FilterSpec, RangeFilter, CountResult, ManifestRow
  client.py     — HTTP client wrapping the two API endpoints
scripts/
  build_index.py          — One-time ASE-DB scan → index.parquet (runs on Eagle)
  estimate_file_sizes.py  — File size calibration: sample → collect → backfill
```

## Architecture

Three layers:
1. **Index** — `index.parquet` (4M rows, one per structure) stored in the `spyhop-index` Modal Volume. Built once from the OMol25 ASE-DB, backfilled with estimated file sizes via n_basis regression.
2. **API** — `app.py` deploys as a Modal app (`modal deploy app.py`). On startup the container mounts the volume and initializes a DuckDB executor. Exposes `POST /query/count` and `POST /query/manifest`.
3. **Clients** — Python library (`spyhop.client`), CLI (`spyhop count` / `spyhop manifest`), static frontend (`index.html`).

The `FilterSpec` dataclass in `query.py` is the shared contract between all clients and the server. Pydantic models for FastAPI validation live in `app.py` only (server concern).

## Key Conventions

- Run Python with `uv run`, never bare `python`
- Package dependencies (`pyproject.toml`) are only what the CLI and client need at runtime: `httpx`, `typer`, `pydantic`. DuckDB and PyArrow are server-only and installed by the Modal image.
- The Modal environment for local dev is `dev` (`garden-ai` account). The `MODAL_ENV` env var switches between `dev` and `main`.
- The API base URL is hardcoded in `client.py` and `index.html` but overridable via `SPYHOP_URL` env var.
- Domain values are validated against a fixed set of 10 in `query.py` (`VALID_DOMAINS`). If the index is extended to OMol-1 or OPoly26, update this set.
