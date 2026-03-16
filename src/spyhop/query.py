"""Domain types shared by all spyhop clients and the server.

FilterSpec is the contract between all clients and the backend — a plain
JSON-serializable description of what the user wants.  The DuckDB executor
that translates it to a query lives in app.py (server-only).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Filter spec ────────────────────────────────────────────────────────

VALID_DOMAINS = {
    "biomolecules", "elytes", "metal_complexes", "reactivity",
    "ani2x", "trans1x", "geom_orca6", "rgd", "orbnet_denali", "spice",
}

VALID_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}


@dataclass
class RangeFilter:
    min: int | float | None = None
    max: int | float | None = None


@dataclass
class FilterSpec:
    must_have: list[str] = field(default_factory=list)
    must_not_have: list[str] = field(default_factory=list)
    domain: str | None = None
    num_atoms: RangeFilter | None = None
    charge: int | None = None
    spin: int | None = None

    def __post_init__(self) -> None:
        for sym in self.must_have + self.must_not_have:
            if sym not in VALID_ELEMENTS:
                raise ValueError(f"Unknown element: {sym!r}")
        overlap = set(self.must_have) & set(self.must_not_have)
        if overlap:
            raise ValueError(f"Elements in both must_have and must_not_have: {overlap}")
        if self.domain is not None and self.domain not in VALID_DOMAINS:
            raise ValueError(f"Unknown domain: {self.domain!r}. Valid: {sorted(VALID_DOMAINS)}")

    @classmethod
    def from_dict(cls, d: dict) -> FilterSpec:
        num_atoms = d.get("num_atoms")
        return cls(
            must_have=d.get("must_have", []),
            must_not_have=d.get("must_not_have", []),
            domain=d.get("domain"),
            num_atoms=RangeFilter(**num_atoms) if num_atoms else None,
            charge=d.get("charge"),
            spin=d.get("spin"),
        )


# ── Result types ───────────────────────────────────────────────────────


@dataclass
class CountResult:
    count: int
    estimated_bytes: int

    @property
    def estimated_gb(self) -> float:
        return self.estimated_bytes / 1e9


@dataclass
class ManifestRow:
    eagle_path: str
    size_orca_tar_zst: int
    size_gbw: int
    size_density_mat_npz: int
