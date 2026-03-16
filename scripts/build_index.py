# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "fairchem-core",
#     "pyarrow>=19.0",
#     "ase",
# ]
#
# ///
"""Build the spyhop parquet index from the OMol25 ASE-DB (train_4M split).

Iterates every entry in the ASE-DB, extracts element-presence booleans,
scalar properties, provenance/routing columns, and writes the result to
a single parquet file.  File sizes are left null — they are backfilled
separately via the n_basis regression calibration step.

Usage:
    uv run spyhop/scripts/build_index.py /path/to/train_4M/ -o omol_index.parquet

The ASE-DB path should be the directory containing the .db files from
the train_4M.tar.gz archive (downloaded from HuggingFace).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── All elements we track (Z=1..118) ──────────────────────────────────
# We generate column names like has_H, has_He, ... has_Og.
# Using ase's chemical_symbols list for canonical ordering.
from ase.data import chemical_symbols  # noqa: E402

# chemical_symbols[0] is 'X' (dummy), real elements start at index 1
ELEMENTS = chemical_symbols[1:119]  # H(1) through Og(118)
ELEMENT_SET = set(ELEMENTS)

# ── Composition string parser ─────────────────────────────────────────
# e.g. "B1Br1C27H36N2O16S1" → {"B", "Br", "C", "H", "N", "O", "S"}
_COMP_RE = re.compile(r"([A-Z][a-z]?)(\d+)")


def parse_composition(comp: str) -> set[str]:
    """Extract the set of element symbols from a composition string."""
    return {m.group(1) for m in _COMP_RE.finditer(comp)}


# ── Subsampling tag extraction ────────────────────────────────────────
# Top-level dir for non-omol paths; omol/<second-component> for omol paths.
# Yields 33 distinct tags matching the taxonomy in ase-db-exploration.


def subsampling_tag(source_path: str) -> str:
    parts = source_path.split("/")
    if parts[0] == "omol" and len(parts) > 1:
        return f"omol/{parts[1]}"
    return parts[0]


# ── Schema definition ─────────────────────────────────────────────────
def build_schema() -> pa.Schema:
    fields = []

    # Element presence booleans
    for elem in ELEMENTS:
        fields.append(pa.field(f"has_{elem}", pa.bool_()))

    # Scalar properties from atoms.info
    fields.extend([
        pa.field("charge", pa.int16()),
        pa.field("spin", pa.int16()),
        pa.field("num_atoms", pa.int32()),
        pa.field("num_electrons", pa.int32()),
        pa.field("n_basis", pa.int32()),
        pa.field("n_scf_steps", pa.int32()),
        pa.field("unrestricted", pa.bool_()),
        pa.field("homo_lumo_gap", pa.float32()),
    ])

    # Provenance / routing
    fields.extend([
        pa.field("domain", pa.dictionary(pa.int8(), pa.utf8())),
        pa.field("subsampling", pa.dictionary(pa.int8(), pa.utf8())),
        pa.field("eagle_path", pa.utf8()),
    ])

    # File sizes — nullable, backfilled later via n_basis regression
    fields.extend([
        pa.field("size_orca_tar_zst", pa.int64()),
        pa.field("size_gbw", pa.int64()),
        pa.field("size_density_mat_npz", pa.int64()),
    ])

    return pa.schema(fields)


# ── Main extraction loop ──────────────────────────────────────────────
BATCH_SIZE = 50_000  # rows per record batch before flushing


def extract_index(db_path: str, output_path: str) -> None:
    from fairchem.core.datasets import AseDBDataset

    log.info("Loading ASE-DB from %s", db_path)
    dataset = AseDBDataset({"src": db_path})
    total = len(dataset)
    log.info("Dataset contains %d entries", total)

    schema = build_schema()
    writer = pq.ParquetWriter(
        output_path,
        schema,
        compression="zstd",
        use_dictionary=True,
    )

    # Pre-allocate column buffers
    def make_buffers() -> dict:
        bufs: dict = defaultdict(list)
        return bufs

    bufs = make_buffers()
    written = 0
    t0 = time.time()

    for idx in range(total):
        atoms = dataset.get_atoms(idx)
        info = atoms.info

        # Element presence
        source_path = info.get("source", "")
        composition = info.get("composition", "")
        if composition:
            elements_present = parse_composition(composition)
        else:
            # Fallback: derive from atomic numbers
            elements_present = {chemical_symbols[z] for z in atoms.get_atomic_numbers()}

        for elem in ELEMENTS:
            bufs[f"has_{elem}"].append(elem in elements_present)

        # Scalar properties
        bufs["charge"].append(info.get("charge"))
        bufs["spin"].append(info.get("spin"))
        bufs["num_atoms"].append(info.get("num_atoms"))
        bufs["num_electrons"].append(info.get("num_electrons"))
        bufs["n_basis"].append(info.get("n_basis"))
        bufs["n_scf_steps"].append(info.get("n_scf_steps"))
        bufs["unrestricted"].append(info.get("unrestricted"))

        # homo_lumo_gap may be an array; take the scalar value
        hlg = info.get("homo_lumo_gap")
        if hasattr(hlg, "__len__"):
            hlg = float(hlg[0]) if len(hlg) > 0 else None
        elif hlg is not None:
            hlg = float(hlg)
        bufs["homo_lumo_gap"].append(hlg)

        # Provenance
        bufs["domain"].append(info.get("data_id", "unknown"))
        bufs["subsampling"].append(subsampling_tag(source_path))
        bufs["eagle_path"].append(os.path.dirname(source_path))

        # File sizes — null for now
        bufs["size_orca_tar_zst"].append(None)
        bufs["size_gbw"].append(None)
        bufs["size_density_mat_npz"].append(None)

        # Flush batch
        if (idx + 1) % BATCH_SIZE == 0:
            _flush_batch(writer, schema, bufs)
            written += BATCH_SIZE
            elapsed = time.time() - t0
            rate = written / elapsed
            eta = (total - written) / rate if rate > 0 else 0
            log.info(
                "Wrote %d / %d (%.1f%%) — %.0f rows/s — ETA %.0fm",
                written, total, 100 * written / total, rate, eta / 60,
            )
            bufs = make_buffers()

    # Final partial batch
    remaining = total - written
    if remaining > 0:
        _flush_batch(writer, schema, bufs)
        written += remaining
        log.info("Wrote final batch: %d / %d", written, total)

    writer.close()
    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / 1e6
    log.info(
        "Done. %d rows written to %s (%.1f MB) in %.1f minutes",
        written, output_path, size_mb, elapsed / 60,
    )


def _flush_batch(
    writer: pq.ParquetWriter,
    schema: pa.Schema,
    bufs: dict,
) -> None:
    arrays = []
    for field in schema:
        col = bufs[field.name]
        arrays.append(pa.array(col, type=field.type))
    batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
    writer.write_batch(batch)


# ── CLI ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the spyhop parquet index from the OMol25 ASE-DB.",
    )
    parser.add_argument(
        "db_path",
        help="Path to the train_4M/ directory containing .db files",
    )
    parser.add_argument(
        "-o", "--output",
        default="omol_index.parquet",
        help="Output parquet file path (default: omol_index.parquet)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.db_path):
        log.error("ASE-DB path does not exist or is not a directory: %s", args.db_path)
        sys.exit(1)

    extract_index(args.db_path, args.output)


if __name__ == "__main__":
    main()
