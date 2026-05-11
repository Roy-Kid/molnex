"""Shared fixtures for ``WaterLESSource`` / ``ChargedDimersSource`` tests.

Both fixtures write extended-XYZ files to a ``tmp_path`` directory in the
exact format ASE expects (``ase.io.read(..., format="extxyz", index=":")``):
a comment line carrying ``Lattice="..."``, ``Properties=...``, ``energy=...``
(eV) and ``pbc="T T T"``, then atom rows ``<symbol> x y z fx fy fz``.

The water fixture is a 4-frame trajectory with 2 H₂O per frame in a cubic
cell — small enough to keep the test suite hermetic, large enough to verify
the ``(N, 3)`` shape and ``(3, 3)`` cell contracts.

The dimer fixture writes one extxyz per (class, split) pair with frames
hand-placed at known fragment separations so the
``test_distribution_shift`` invariant is verifiable by construction.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import pytest

# Six canonical dimer classes (kept in lock-step with
# ``charged_dimers._DIMER_CLASSES``).
_DIMER_CLASSES = (
    "C3N3H10+",
    "C2O2H3-",
    "C3H4N+",
    "C2H7N2+",
    "C2HO2-",
    "CH4N-",
)


def _format_lattice(edge: float) -> str:
    """Return a 9-float row-major cubic lattice string for extxyz."""
    return f'"{edge:.6f} 0.0 0.0 0.0 {edge:.6f} 0.0 0.0 0.0 {edge:.6f}"'


def _make_water_frame(
    *,
    energy: float,
    edge: float,
    o_positions: Iterable[tuple[float, float, float]],
    seed: int,
) -> str:
    """Build a single extxyz frame of n_water H₂O molecules.

    Each O is placed at the supplied position; the two H atoms of that
    water hang off the O at canonical OH bond length / HOH angle, with a
    deterministic seed-based jitter so the dataset isn't identical
    between frames. Forces are seed-derived deterministic floats.
    """
    import math
    import random

    rng = random.Random(seed)
    o_list = list(o_positions)
    n_atoms = 3 * len(o_list)
    bond = 0.96  # Å
    half_angle = math.radians(104.5 / 2.0)
    rows: list[str] = []
    fx_total = 0.0  # forces will be neutralized at the end so they sum near zero
    for ox, oy, oz in o_list:
        # O
        fx, fy, fz = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
        rows.append(f"O  {ox:.6f} {oy:.6f} {oz:.6f}  {fx:.6f} {fy:.6f} {fz:.6f}")
        # H₁: along +x rotated by half_angle in xy
        h1x = ox + bond * math.cos(half_angle)
        h1y = oy + bond * math.sin(half_angle)
        h1z = oz
        fx, fy, fz = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
        rows.append(f"H  {h1x:.6f} {h1y:.6f} {h1z:.6f}  {fx:.6f} {fy:.6f} {fz:.6f}")
        # H₂
        h2x = ox + bond * math.cos(half_angle)
        h2y = oy - bond * math.sin(half_angle)
        h2z = oz
        fx, fy, fz = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
        rows.append(f"H  {h2x:.6f} {h2y:.6f} {h2z:.6f}  {fx:.6f} {fy:.6f} {fz:.6f}")
        fx_total += 1
    comment = (
        f"Lattice={_format_lattice(edge)} "
        f"Properties=species:S:1:pos:R:3:forces:R:3 "
        f"energy={energy:.6f} "
        f'pbc="T T T"'
    )
    return "\n".join([str(n_atoms), comment, *rows]) + "\n"


#: Per-atom binding-energy reference used to build paper-realistic fixture
#: energies. RPBE-D3 bulk water sits near ``-10 eV/atom`` (cf. Cheng 2025,
#: npj Comput. Mater. 11:80 — 64 H₂O / 192 atoms total energies cluster
#: around ``-2000 eV``). A 2-water 6-atom micro-fixture therefore targets
#: ``-60 eV`` ± a small per-frame jitter, keeping the fixture in the same
#: order of magnitude as the upstream extxyz files so unit checks are
#: meaningful.
_WATER_E0_PER_ATOM_EV: float = -10.0


def _write_water_extxyz(path: Path, *, n_frames: int, n_water: int, edge: float) -> None:
    """Write ``n_frames`` water-box frames to *path*.

    Each frame's O atoms sit on a deterministic 1-D string spaced by
    ``edge / (n_water + 1)`` so the box is well within periodic limits
    and atoms never fall outside the cubic cell. Per-frame energies follow
    ``E(i) = (3 · n_water) · _WATER_E0_PER_ATOM_EV - 0.1·i`` so the values
    land in the eV-scale range RPBE-D3 actually produces (≈ ``-10 eV/atom``);
    forces are ``[-1, 1] eV·Å⁻¹`` (well below the 50 eV·Å⁻¹ sustained-force
    threshold typical of well-converged liquid-water frames).
    """
    spacing = edge / (n_water + 1)
    o_positions = [((i + 1) * spacing, edge / 2.0, edge / 2.0) for i in range(n_water)]
    n_atoms = 3 * n_water
    e_base = n_atoms * _WATER_E0_PER_ATOM_EV
    frames = [
        _make_water_frame(
            energy=e_base - 0.1 * frame_i,
            edge=edge,
            o_positions=o_positions,
            seed=frame_i,
        )
        for frame_i in range(n_frames)
    ]
    path.write_text("".join(frames))


def _make_dimer_frame(
    *,
    energy: float,
    edge: float,
    separation: float,
    seed: int,
) -> str:
    """Build a single 2-atom dimer frame at fragment separation ``separation``.

    Fragment 1 sits at the origin; fragment 2 sits ``(separation, 0, 0)``
    away. With one atom per fragment, the inter-fragment min distance is
    exactly ``separation``. This keeps the
    ``ChargedDimersSource._fragment_separation`` extractor's geometric
    contract verifiable without depending on real chemistry.
    """
    import random

    rng = random.Random(seed)
    cx, cy, cz = edge / 2.0, edge / 2.0, edge / 2.0
    # Fragment 1: C at centre.
    f1x, f1y, f1z = cx, cy, cz
    # Fragment 2: N at (cx + separation, cy, cz).
    f2x, f2y, f2z = cx + separation, cy, cz
    rows = [
        (
            f"C  {f1x:.6f} {f1y:.6f} {f1z:.6f}  "
            f"{rng.uniform(-1.0, 1.0):.6f} {rng.uniform(-1.0, 1.0):.6f} "
            f"{rng.uniform(-1.0, 1.0):.6f}"
        ),
        (
            f"N  {f2x:.6f} {f2y:.6f} {f2z:.6f}  "
            f"{rng.uniform(-1.0, 1.0):.6f} {rng.uniform(-1.0, 1.0):.6f} "
            f"{rng.uniform(-1.0, 1.0):.6f}"
        ),
    ]
    comment = (
        f"Lattice={_format_lattice(edge)} "
        f"Properties=species:S:1:pos:R:3:forces:R:3 "
        f"energy={energy:.6f} "
        f'pbc="T T T"'
    )
    return "\n".join(["2", comment, *rows]) + "\n"


def _write_dimer_extxyz(path: Path, *, separations: Iterable[float], edge: float) -> None:
    """Write one extxyz with one 2-atom dimer frame per supplied separation."""
    seps = list(separations)
    frames = [
        _make_dimer_frame(
            energy=-10.0 - 0.05 * i,
            edge=edge,
            separation=s,
            seed=i,
        )
        for i, s in enumerate(seps)
    ]
    path.write_text("".join(frames))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def water_les_root(tmp_path: Path) -> Path:
    """Directory with synthetic ``train-H2O_RPBE-D3.xyz`` (40 frames) and
    ``test-H2O_RPBE-D3.xyz`` (2 frames).

    40 train frames is the smallest count that produces a clean
    0.95 / 0.05 deterministic slice (38 train / 2 val) so the
    deterministic-split test asserts disjoint, ordered, non-empty
    slices on both sides.
    """
    root = tmp_path / "water_les"
    root.mkdir()
    train_path = root / "train-H2O_RPBE-D3.xyz"
    test_path = root / "test-H2O_RPBE-D3.xyz"
    _write_water_extxyz(train_path, n_frames=40, n_water=2, edge=12.0)
    _write_water_extxyz(test_path, n_frames=2, n_water=2, edge=12.0)
    return root


@pytest.fixture
def water_les_checksums(water_les_root: Path) -> dict[str, str]:
    """SHA-256 of every file under ``water_les_root``.

    Useful to monkey-patch ``WaterLESSource._CHECKSUMS`` and exercise
    the ``verify_checksum=True`` branch deterministically.
    """
    return {
        "train-H2O_RPBE-D3.xyz": _sha256(water_les_root / "train-H2O_RPBE-D3.xyz"),
        "test-H2O_RPBE-D3.xyz": _sha256(water_les_root / "test-H2O_RPBE-D3.xyz"),
    }


@pytest.fixture
def charged_dimers_root(tmp_path: Path) -> Path:
    """Directory with synthetic per-class extxyz files for all 6 dimer classes.

    Each class gets ``train.xyz`` with three separations (6.0, 8.0, 11.5 Å —
    deliberately unsorted so the source's sort-by-separation invariant
    is testable) and ``test.xyz`` with two separations (12.5 and 14.0 Å).
    """
    root = tmp_path / "charged_dimers"
    root.mkdir()
    for cls in _DIMER_CLASSES:
        cls_dir = root / cls
        cls_dir.mkdir()
        # Unsorted on purpose — Source must sort.
        _write_dimer_extxyz(
            cls_dir / "train.xyz",
            separations=[8.0, 6.0, 11.5],
            edge=30.0,
        )
        _write_dimer_extxyz(
            cls_dir / "test.xyz",
            separations=[14.0, 12.5],
            edge=30.0,
        )
    return root


@pytest.fixture
def dimer_classes() -> tuple[str, ...]:
    return _DIMER_CLASSES
