# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "pyarrow>=19.0",
#     "numpy",
#     "globus-sdk>=3.0",
# ]
#
# ///
"""Estimate per-structure file sizes and backfill the spyhop parquet index.

This script implements the n_basis regression strategy described in the design
doc.  It operates in three modes:

1. **sample** — Select ~200 paths stratified by domain × n_basis quartile from
   the index, and write them to a CSV with empty size columns.

2. **collect** — Read the sample CSV, query the Eagle Globus collection for
   actual file sizes via the Globus SDK, and write a filled CSV.

3. **backfill** — Read the filled CSV, fit regression models on n_basis, apply
   them to all rows, and write an updated parquet file.

Usage:
    # Step 1: generate the sample paths
    uv run spyhop/scripts/estimate_file_sizes.py sample \
        spyhop/index.parquet -o spyhop/size_sample_paths.csv

    # Step 2: collect actual sizes from Globus
    uv run spyhop/scripts/estimate_file_sizes.py collect \
        spyhop/size_sample_paths.csv -o spyhop/size_sample_filled.csv

    # Step 3: backfill the index
    uv run spyhop/scripts/estimate_file_sizes.py backfill \
        spyhop/index.parquet spyhop/size_sample_filled.csv \
        -o spyhop/index_with_sizes.parquet
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Number of samples per stratum (domain × n_basis quartile)
SAMPLES_PER_STRATUM = 5

# Eagle Globus collection endpoint
EAGLE_COLLECTION_ID = "0b73865a-ff20-4f57-a1d7-573d86b54624"

# Spyhop Globus native app client ID (registered at app.globus.org)
GLOBUS_CLIENT_ID = "51ae67ff-385d-4633-a337-e432a25bd76f"

# Expected files per structure directory
FILE_MAP = {
    "orca.tar.zst": "size_orca_tar_zst",
    "orca.gbw.zstd0": "size_gbw",
    "density_mat.npz": "size_density_mat_npz",
}


# ── sample mode ───────────────────────────────────────────────────────


def cmd_sample(args: argparse.Namespace) -> None:
    """Select stratified sample paths and write to CSV."""
    log.info("Reading index from %s", args.index)
    table = pq.read_table(args.index, columns=["eagle_path", "n_basis", "n_scf_steps", "domain"])

    eagle_path = table.column("eagle_path").to_pylist()
    n_basis = table.column("n_basis").to_numpy()
    n_scf_steps = table.column("n_scf_steps").to_numpy()
    domain = table.column("domain").to_pylist()

    # Compute global n_basis quartile boundaries
    quartiles = np.quantile(n_basis, [0.25, 0.5, 0.75])
    log.info("n_basis quartile boundaries: %s", quartiles)

    def quartile_bin(val: int) -> int:
        for i, q in enumerate(quartiles):
            if val <= q:
                return i
        return len(quartiles)

    # Group indices by (domain, quartile)
    strata: dict[tuple[str, int], list[int]] = {}
    for i in range(len(eagle_path)):
        key = (domain[i], quartile_bin(n_basis[i]))
        strata.setdefault(key, []).append(i)

    # Sample from each stratum
    rng = np.random.default_rng()
    selected: list[int] = []
    for key, indices in sorted(strata.items()):
        n = min(SAMPLES_PER_STRATUM, len(indices))
        chosen = rng.choice(indices, size=n, replace=False)
        selected.extend(chosen)
        log.info("Stratum %s: %d available, sampled %d", key, len(indices), n)

    selected.sort()
    log.info("Total samples selected: %d", len(selected))

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "eagle_path", "n_basis", "n_scf_steps", "domain",
            "size_orca_tar_zst", "size_gbw", "size_density_mat_npz",
        ])
        for i in selected:
            writer.writerow([
                eagle_path[i], int(n_basis[i]), int(n_scf_steps[i]), domain[i],
                "",  # size_orca_tar_zst — fill in manually
                "",  # size_gbw — fill in manually
                "",  # size_density_mat_npz — fill in manually
            ])

    log.info("Wrote %d sample paths to %s", len(selected), args.output)
    log.info("Next: run `collect` to query actual file sizes from Globus.")


# ── collect mode ──────────────────────────────────────────────────────


def _get_transfer_client() -> "globus_sdk.TransferClient":
    """Authenticate via Globus native app flow and return a TransferClient."""
    import globus_sdk

    client = globus_sdk.NativeAppAuthClient(GLOBUS_CLIENT_ID)
    client.oauth2_start_flow(
        requested_scopes=globus_sdk.TransferClient.scopes.all,
    )

    authorize_url = client.oauth2_get_authorize_url()
    print(f"\nPlease visit this URL to authenticate:\n\n  {authorize_url}\n")
    auth_code = input("Enter the authorization code: ").strip()

    tokens = client.oauth2_exchange_code_for_tokens(auth_code)
    transfer_tokens = tokens.by_resource_server["transfer.api.globus.org"]
    authorizer = globus_sdk.AccessTokenAuthorizer(
        transfer_tokens["access_token"],
    )
    return globus_sdk.TransferClient(authorizer=authorizer)


def cmd_collect(args: argparse.Namespace) -> None:
    """Query Globus for actual file sizes and fill in the sample CSV."""
    # Read existing sample CSV
    rows = []
    with open(args.input_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    log.info("Loaded %d sample paths from %s", len(rows), args.input_csv)

    # Authenticate
    tc = _get_transfer_client()
    log.info("Authenticated with Globus. Querying file sizes...")

    succeeded = 0
    failed = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        eagle_path = row["eagle_path"]
        try:
            ls_result = tc.operation_ls(
                EAGLE_COLLECTION_ID,
                path=f"/{eagle_path}/",
            )
            # Build name → size map from directory listing
            file_sizes = {
                entry["name"]: entry["size"]
                for entry in ls_result
                if entry["type"] == "file"
            }
            # Match expected files
            for filename, col in FILE_MAP.items():
                if filename in file_sizes:
                    row[col] = str(file_sizes[filename])
                else:
                    log.warning(
                        "  [%d] %s: missing %s", i, eagle_path, filename,
                    )
            succeeded += 1
        except Exception as e:
            log.warning("  [%d] %s: %s", i, eagle_path, e)
            failed += 1

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(rows) - i - 1) / rate if rate > 0 else 0
            log.info(
                "  Progress: %d / %d (%.0f/s, ETA %.0fs)",
                i + 1, len(rows), rate, eta,
            )

    elapsed = time.time() - t0
    log.info(
        "Done: %d succeeded, %d failed out of %d (%.1fs)",
        succeeded, failed, len(rows), elapsed,
    )

    # Write filled CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote filled CSV to %s", args.output)


# ── backfill mode ─────────────────────────────────────────────────────


def _fit_power_model(
    x: np.ndarray,
    y: np.ndarray,
    label: str,
) -> tuple[float, float]:
    """Fit y = a * x^b in log-log space via least squares.

    Returns (a, b).  Falls back to y = a * x^2 (fixed b=2) if the fit
    produces an unreasonable exponent, since density_mat and gbw are
    theoretically quadratic in n_basis.
    """
    mask = (x > 0) & (y > 0)
    lx = np.log(x[mask])
    ly = np.log(y[mask])

    # Fit log(y) = log(a) + b*log(x)
    coeffs = np.polyfit(lx, ly, 1)
    b, log_a = coeffs
    a = np.exp(log_a)

    # Residual stats
    predicted = a * x[mask] ** b
    residuals = (y[mask] - predicted) / y[mask]
    mape = np.mean(np.abs(residuals)) * 100

    log.info(
        "  %s: y = %.4e * x^%.3f  (MAPE=%.1f%%, n=%d)",
        label, a, b, mape, mask.sum(),
    )
    return a, b


def cmd_backfill(args: argparse.Namespace) -> None:
    """Fit regression from calibration CSV, apply to full index."""
    # Read calibration data
    log.info("Reading calibration data from %s", args.calibration_csv)
    cal_n_basis = []
    cal_n_scf_steps = []
    cal_sizes: dict[str, list[float]] = {
        "size_orca_tar_zst": [],
        "size_gbw": [],
        "size_density_mat_npz": [],
    }

    with open(args.calibration_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows where sizes haven't been filled in
            if not row["size_orca_tar_zst"] or not row["size_gbw"] or not row["size_density_mat_npz"]:
                continue
            cal_n_basis.append(int(row["n_basis"]))
            cal_n_scf_steps.append(int(row["n_scf_steps"]))
            for col in cal_sizes:
                cal_sizes[col].append(float(row[col]))

    cal_n_basis_arr = np.array(cal_n_basis, dtype=np.float64)
    cal_n_scf_steps_arr = np.array(cal_n_scf_steps, dtype=np.float64)
    n_cal = len(cal_n_basis)
    log.info("Loaded %d calibration points", n_cal)

    if n_cal < 10:
        log.error("Need at least 10 calibration points, got %d", n_cal)
        sys.exit(1)

    # Fit models
    log.info("Fitting regression models:")
    models: dict[str, tuple[float, float]] = {}

    # density_mat and gbw: expect ~quadratic in n_basis
    for col in ["size_density_mat_npz", "size_gbw"]:
        a, b = _fit_power_model(cal_n_basis_arr, np.array(cal_sizes[col]), col)
        models[col] = (a, b)

    # orca.tar.zst: fit on n_basis * n_scf_steps as the composite predictor
    # (text output volume scales with both system size and convergence steps)
    composite = cal_n_basis_arr * cal_n_scf_steps_arr
    a, b = _fit_power_model(
        composite, np.array(cal_sizes["size_orca_tar_zst"]), "size_orca_tar_zst",
    )
    models["size_orca_tar_zst"] = (a, b)
    orca_uses_composite = True

    # Read full index
    log.info("Reading full index from %s", args.index)
    table = pq.read_table(args.index)
    n_rows = table.num_rows
    log.info("Index has %d rows", n_rows)

    n_basis_full = table.column("n_basis").to_numpy().astype(np.float64)
    n_scf_steps_full = table.column("n_scf_steps").to_numpy().astype(np.float64)

    # Apply models
    log.info("Applying models to %d rows...", n_rows)
    new_columns: dict[str, np.ndarray] = {}

    for col in ["size_density_mat_npz", "size_gbw"]:
        a, b = models[col]
        new_columns[col] = np.round(a * n_basis_full**b).astype(np.int64)

    a, b = models["size_orca_tar_zst"]
    composite_full = n_basis_full * n_scf_steps_full
    new_columns["size_orca_tar_zst"] = np.round(a * composite_full**b).astype(np.int64)

    # Report aggregate stats
    total_bytes = sum(new_columns[c].sum() for c in new_columns)
    log.info(
        "Estimated total dataset size: %.1f TB",
        total_bytes / 1e12,
    )
    for col in new_columns:
        arr = new_columns[col]
        log.info(
            "  %s: median=%.1f MB, mean=%.1f MB, total=%.1f TB",
            col, np.median(arr) / 1e6, np.mean(arr) / 1e6, arr.sum() / 1e12,
        )

    # Replace the size columns in the table
    schema = table.schema
    col_names = [f.name for f in schema]
    arrays = []
    for name in col_names:
        if name in new_columns:
            arrays.append(pa.array(new_columns[name], type=pa.int64()))
        else:
            arrays.append(table.column(name))

    new_table = pa.table(dict(zip(col_names, arrays)), schema=schema)

    # Write output
    output = args.output
    pq.write_table(
        new_table,
        output,
        compression="zstd",
        use_dictionary=True,
    )

    size_mb = os.path.getsize(output) / 1e6
    log.info("Wrote updated index to %s (%.1f MB)", output, size_mb)


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate file sizes for the spyhop parquet index.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # sample subcommand
    p_sample = sub.add_parser("sample", help="Generate stratified sample paths for size collection")
    p_sample.add_argument("index", help="Path to the spyhop index parquet file")
    p_sample.add_argument("-o", "--output", default="size_sample_paths.csv", help="Output CSV path")

    # collect subcommand
    p_collect = sub.add_parser("collect", help="Query Globus for actual file sizes")
    p_collect.add_argument("input_csv", help="CSV from the sample step (with empty size columns)")
    p_collect.add_argument("-o", "--output", default="size_sample_filled.csv", help="Output CSV path")

    # backfill subcommand
    p_backfill = sub.add_parser("backfill", help="Fit regression and backfill file sizes in the index")
    p_backfill.add_argument("index", help="Path to the spyhop index parquet file")
    p_backfill.add_argument("calibration_csv", help="CSV with eagle_path and actual file sizes")
    p_backfill.add_argument("-o", "--output", default="index_with_sizes.parquet", help="Output parquet path")

    args = parser.parse_args()

    if args.command == "sample":
        cmd_sample(args)
    elif args.command == "collect":
        cmd_collect(args)
    elif args.command == "backfill":
        cmd_backfill(args)


if __name__ == "__main__":
    main()
