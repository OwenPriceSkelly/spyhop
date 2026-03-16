from __future__ import annotations

from typing import Annotated, Optional

import typer

from spyhop import client
from spyhop.query import FilterSpec, RangeFilter

cli = typer.Typer(name="spyhop", help="Query the OMol25 filter index.")


def _build_spec(
    has: list[str],
    not_has: list[str],
    domain: Optional[str],
    min_atoms: Optional[int],
    max_atoms: Optional[int],
    charge: Optional[int],
    spin: Optional[int],
) -> FilterSpec:
    num_atoms = None
    if min_atoms is not None or max_atoms is not None:
        num_atoms = RangeFilter(min=min_atoms, max=max_atoms)
    return FilterSpec(
        must_have=has,
        must_not_have=not_has,
        domain=domain,
        num_atoms=num_atoms,
        charge=charge,
        spin=spin,
    )


@cli.command()
def count(
    has: Annotated[Optional[list[str]], typer.Option("--has", metavar="ELEMENT", help="Must contain element")] = None,
    not_has: Annotated[Optional[list[str]], typer.Option("--not", metavar="ELEMENT", help="Must not contain element")] = None,
    domain: Annotated[Optional[str], typer.Option(help="Filter by domain (e.g. elytes, biomolecules)")] = None,
    min_atoms: Annotated[Optional[int], typer.Option(help="Minimum number of atoms")] = None,
    max_atoms: Annotated[Optional[int], typer.Option(help="Maximum number of atoms")] = None,
    charge: Annotated[Optional[int], typer.Option(help="Formal charge")] = None,
    spin: Annotated[Optional[int], typer.Option(help="Spin multiplicity (2S+1)")] = None,
    url: Annotated[Optional[str], typer.Option("--url", help="Override the API base URL", envvar="SPYHOP_URL")] = None,
):
    """Count matching structures and estimate transfer size."""
    kwargs = {"base_url": url} if url else {}
    spec = _build_spec(has or [], not_has or [], domain, min_atoms, max_atoms, charge, spin)
    result = client.count(spec, **kwargs)
    typer.echo(f"{result.count:,} structures (~{result.estimated_gb:.1f} GB)")


@cli.command()
def manifest(
    has: Annotated[Optional[list[str]], typer.Option("--has", metavar="ELEMENT", help="Must contain element")] = None,
    not_has: Annotated[Optional[list[str]], typer.Option("--not", metavar="ELEMENT", help="Must not contain element")] = None,
    domain: Annotated[Optional[str], typer.Option(help="Filter by domain (e.g. elytes, biomolecules)")] = None,
    min_atoms: Annotated[Optional[int], typer.Option(help="Minimum number of atoms")] = None,
    max_atoms: Annotated[Optional[int], typer.Option(help="Maximum number of atoms")] = None,
    charge: Annotated[Optional[int], typer.Option(help="Formal charge")] = None,
    spin: Annotated[Optional[int], typer.Option(help="Spin multiplicity (2S+1)")] = None,
    output: Annotated[str, typer.Option("-o", "--output", help="Output manifest path")] = "manifest.txt",
    url: Annotated[Optional[str], typer.Option("--url", help="Override the API base URL", envvar="SPYHOP_URL")] = None,
):
    """Write a Globus transfer manifest for matching structures."""
    kwargs = {"base_url": url} if url else {}
    spec = _build_spec(has or [], not_has or [], domain, min_atoms, max_atoms, charge, spin)
    rows = client.manifest(spec, **kwargs)
    _write_globus_batch(rows, output)
    typer.echo(f"Wrote {len(rows):,} paths to {output}")
    typer.echo(f"Usage: globus transfer $EAGLE_EP:/ $DEST_EP:/local/path/ --batch {output} --recursive")


def _write_globus_batch(rows: list, output_path: str) -> None:
    import shlex
    with open(output_path, "w") as f:
        f.write(f"# OMol25 subset — {len(rows):,} systems\n")
        f.write("# Usage: globus transfer $EAGLE_EP:/ $DEST_EP:/local/path/ --batch manifest.txt --recursive\n")
        for row in rows:
            path = row.eagle_path.rstrip("/") + "/"
            quoted = shlex.quote(path)
            f.write(f"{quoted} {quoted}\n")


def main() -> None:
    cli()
