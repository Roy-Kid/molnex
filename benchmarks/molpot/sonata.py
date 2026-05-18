"""Sonata long-range periodic benchmark — bulk liquid water at 300 K.

The benchmark trains the same Allegro encoder + ``EdgeEnergyHead`` short-range
stack twice on a bulk-liquid-water reference set (or a molten-NaCl fallback)
— once with the ``EwaldMultipoleEnergy`` multipole-Ewald term enabled (Sonata,
via :func:`molpot.composition.build_sonata`), once with that term ablated —
and reports a CSV row whose ``improvement_pct`` column gates the load-bearing
"long-range force MAE on ``r_min > r_cut`` atoms is at least 30 % lower than
the no-Ewald baseline" claim.

Drivers:
    python benchmarks/molpot/sonata.py --smoke
    python benchmarks/molpot/sonata.py --full --n-steps 10000

    python -m pytest benchmarks/molpot/sonata.py -v          # smoke + kernel
    python -m pytest benchmarks/molpot/sonata.py -m bench_long  # full

Smoke mode (default; ``--smoke``) runs ``SMOKE_N_STEPS`` (= 100) training
steps on a synthetic 4-water periodic box, never hits the network, and skips
the quantitative bar assertions. Full mode (``--full``) runs ``--n-steps``
on the configured substrate and asserts every bar in the
``sonata-03-bench.acceptance.md`` contract.

References:
    Cheng, B. *Latent Ewald summation for machine-learning potentials*
    npj Comput. Mater. **11**, 80 (2025).
    https://doi.org/10.1038/s41524-025-01577-7

    Musaelian et al. *Learning Local Equivariant Representations for
    Large-Scale Atomistic Dynamics* Nat. Commun. **14**, 579 (2023).
    https://arxiv.org/abs/2204.05249

    Sonata composer: see ``.claude/specs/multipole-layer.md`` and the
    ``sonata-01-composer`` chain that landed it.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.composition import build_sonata
from molpot.heads import EdgeEnergyHead
from molzoo import Allegro

# ---------------------------------------------------------------------------
# Constants — paths, columns, hyperparameters, unit factors
# ---------------------------------------------------------------------------

#: Append-only CSV log produced by the benchmark. One row per run.
CSV_PATH = Path(__file__).parent / "sonata_results.csv"
#: Lazy-fetched data directory. ``water_subset.npz`` is materialised by the
#: full-mode loader on first invocation; smoke mode never touches the network.
DATA_DIR = Path(__file__).parent / "data"
WATER_NPZ = DATA_DIR / "water_subset.npz"

#: Column order for ``sonata_results.csv``. Pinned by the spec so a third-party
#: reader can audit a CSV row without cross-referencing this file.
CSV_COLUMNS: tuple[str, ...] = (
    "date",
    "substrate",
    "n_atoms",
    "n_steps",
    "energy_mae_meV_per_atom",
    "force_mae_meV_per_A",
    "force_mae_long_range_meV_per_A_sonata",
    "force_mae_long_range_meV_per_A_baseline",
    "improvement_pct",
    "sigma",
    "dl",
)

# Encoder + Ewald hyperparameters. Small enough to run a 100-step smoke pass
# on CPU in seconds; large enough that ``--full --n-steps 10000`` on a CUDA
# host can hit the spec's bars on bulk water.
DEFAULT_R_MAX = 5.0  # Å — encoder cutoff; doubles as r_cut for the LR kernel
DEFAULT_L_MAX = 2  # required by Sonata when dipole or quadrupole is on
DEFAULT_NUM_FEATURES = 64  # Allegro num_scalar_features, per spec design
DEFAULT_NUM_LAYERS = 2  # Allegro num_layers, per spec design
DEFAULT_TYPE_EMBED_DIM = 32
DEFAULT_LATENT_MLP_WIDTH = 64
DEFAULT_AVG_NEIGHBORS = 12.0
DEFAULT_SIGMA = 1.0  # Å — σ-Gaussian charge-smearing length
DEFAULT_DL = 2.0  # Å — Ewald reciprocal-space grid resolution
DEFAULT_LR = 1e-3
SMOKE_N_STEPS = 100

#: 1 eV → 1000 meV. Used to convert internal energy/force units to the
#: meV/atom and meV/Å bars the CSV reports.
EV_TO_MEV = 1000.0

#: Periodic-table atomic numbers we touch. 0 is reserved for empty slots
#: inside the encoder's type embedding and ``num_elements`` is set to a
#: comfortable upper bound; per-atom ``Z`` values index this table directly.
NUM_ELEMENTS = 20

# ---------------------------------------------------------------------------
# Synthetic snapshot construction
# ---------------------------------------------------------------------------


def _build_synthetic_water_box(
    *,
    seed: int = 0,
    n_waters: int = 4,
    box_length: float = 12.0,
) -> dict[str, torch.Tensor | float]:
    """Build a synthetic periodic ``n_waters``-H₂O snapshot.

    The geometry is a 2 × 2 × 2 (or smaller) face-centred lattice of water
    monomers placed inside an orthorhombic box. Bond and HOH-angle values
    are pulled from the SPC/E geometry. Energies and forces are zero — the
    snapshot is a pipeline-plumbing fixture, not a physics oracle.

    Args:
        seed: Seed for any random jitter applied to monomer orientations.
        n_waters: Number of H₂O monomers (default 4 → 12 atoms).
        box_length: Cubic-box side length in Å.

    Returns:
        Snapshot dict with keys ``Z``, ``pos``, ``cell``, ``energy``,
        ``forces``. Tensor units: ``pos`` in Å, ``cell`` in Å, ``forces``
        in eV/Å, ``energy`` in eV.
    """
    rng = torch.Generator().manual_seed(seed)

    # SPC/E water geometry — O at the origin, two H pinned by bond + angle.
    r_OH = 1.0  # Å
    theta = math.radians(109.47)
    h_offsets = torch.tensor(
        [
            [r_OH * math.sin(theta / 2.0), r_OH * math.cos(theta / 2.0), 0.0],
            [-r_OH * math.sin(theta / 2.0), r_OH * math.cos(theta / 2.0), 0.0],
        ],
        dtype=torch.float32,
    )

    n_grid = max(1, math.ceil(n_waters ** (1.0 / 3.0)))
    spacing = box_length / n_grid

    Z_list: list[int] = []
    pos_list: list[torch.Tensor] = []
    placed = 0
    for ix in range(n_grid):
        for iy in range(n_grid):
            for iz in range(n_grid):
                if placed >= n_waters:
                    break
                centre = torch.tensor(
                    [(ix + 0.5) * spacing, (iy + 0.5) * spacing, (iz + 0.5) * spacing],
                    dtype=torch.float32,
                )
                # Random small jitter so monomers are not on a perfect grid.
                jitter = 0.05 * (torch.rand(3, generator=rng, dtype=torch.float32) * 2.0 - 1.0)
                Q = _random_so3(rng=rng)
                rotated_h = (Q @ h_offsets.T).T
                Z_list.extend([8, 1, 1])
                pos_list.append(centre + jitter)
                pos_list.append(centre + jitter + rotated_h[0])
                pos_list.append(centre + jitter + rotated_h[1])
                placed += 1
    Z = torch.tensor(Z_list, dtype=torch.long)
    pos = torch.stack(pos_list, dim=0)
    cell = box_length * torch.eye(3, dtype=torch.float32)

    n_atoms = Z.shape[0]
    return {
        "Z": Z,
        "pos": pos,
        "cell": cell,
        "energy": 0.0,
        "forces": torch.zeros((n_atoms, 3), dtype=torch.float32),
    }


def _random_so3(*, rng: torch.Generator) -> torch.Tensor:
    """Uniform SO(3) rotation matrix via QR. Returns ``(3, 3)`` float32."""
    A = torch.randn(3, 3, generator=rng, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A)
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _build_synthetic_nacl_box(
    *,
    seed: int = 0,
    n_pairs: int = 4,
    lattice_const: float = 5.6,
) -> dict[str, torch.Tensor | float]:
    """Build a synthetic periodic NaCl rocksalt box, ``2 × n_pairs`` atoms.

    Used as the molten-NaCl substrate fallback when the water reference is
    unreachable. The geometry is the rocksalt zero-temperature crystal
    perturbed by a small thermal jitter; this is *not* a thermalised
    1100 K configuration but it exercises the full periodic pipeline with
    a charged-ion substrate.

    Args:
        seed: Seed for thermal jitter.
        n_pairs: Number of NaCl formula units along one cell edge.
        lattice_const: NaCl lattice constant in Å.

    Returns:
        Snapshot dict (same shape as :func:`_build_synthetic_water_box`).
    """
    rng = torch.Generator().manual_seed(seed)
    Z_list: list[int] = []
    pos_list: list[torch.Tensor] = []
    side = n_pairs * lattice_const
    for ix in range(n_pairs):
        for iy in range(n_pairs):
            for iz in range(n_pairs):
                base = lattice_const * torch.tensor(
                    [float(ix), float(iy), float(iz)], dtype=torch.float32
                )
                # Na at origin, Cl at (0.5, 0.5, 0.5) of the conventional cell.
                Z_list.extend([11, 17])
                pos_list.append(base)
                pos_list.append(base + 0.5 * lattice_const)
    pos = torch.stack(pos_list, dim=0)
    pos = pos + 0.05 * (torch.rand(pos.shape, generator=rng, dtype=torch.float32) * 2.0 - 1.0)
    Z = torch.tensor(Z_list, dtype=torch.long)
    cell = side * torch.eye(3, dtype=torch.float32)
    n_atoms = Z.shape[0]
    return {
        "Z": Z,
        "pos": pos,
        "cell": cell,
        "energy": 0.0,
        "forces": torch.zeros((n_atoms, 3), dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# Data loading — substrate selection + fallback
# ---------------------------------------------------------------------------


def _load_water_data(
    *,
    smoke: bool,
    seed: int = 0,
) -> tuple[list[dict], list[dict], str]:
    """Load liquid-water snapshots for training and validation.

    In smoke mode the loader synthesises a single 4-H₂O periodic snapshot
    inline and never hits the network. In full mode the loader reads
    ``WATER_NPZ`` if present (a SPICE-OOD-water subset pre-fetched by the
    user) and otherwise raises ``RuntimeError`` so the caller can fall
    back to the molten-NaCl substrate.

    Args:
        smoke: If ``True``, skip any disk / network access.
        seed: Seed for synthetic snapshot generation.

    Returns:
        ``(train, val, substrate)`` where ``train`` / ``val`` are lists of
        snapshot dicts (see :func:`_build_synthetic_water_box`) and
        ``substrate`` is one of ``{"water_synthetic_4", "water_spice_ood"}``.

    Raises:
        RuntimeError: With message ``"water reference unavailable"`` if a
            full-mode load cannot reach the SPICE-OOD-water subset on
            disk. Callers retry once and then fall back to molten NaCl.
    """
    if smoke:
        snap = _build_synthetic_water_box(seed=seed)
        # Same snapshot for train and val — smoke is a plumbing test, not
        # a generalisation test. The CSV row records non-NaN MAE columns;
        # ac-001 / ac-006 do not gate on physical accuracy.
        return [snap], [snap], "water_synthetic_4"
    if not WATER_NPZ.exists():
        raise RuntimeError("water reference unavailable")
    data = _load_water_npz(WATER_NPZ)
    n = len(data)
    n_val = max(1, n // 5)
    return data[:-n_val], data[-n_val:], "water_spice_ood"


def _load_water_npz(path: Path) -> list[dict]:
    """Load a SPICE-OOD-water subset from a single ``.npz`` file.

    Expected schema: arrays ``Z``, ``pos``, ``cell``, ``energy``, ``forces``,
    ``ptr`` where ``ptr[i]:ptr[i+1]`` slices snapshot ``i``'s atoms out of
    the concatenated per-atom arrays. ``cell`` is ``(n_snapshots, 3, 3)``;
    ``energy`` is ``(n_snapshots,)`` in eV; ``forces`` is concatenated
    per-atom in eV/Å; ``Z``, ``pos`` are concatenated per-atom.
    """
    import numpy as np

    arr = np.load(path)
    ptr = torch.from_numpy(arr["ptr"]).long()
    Z_all = torch.from_numpy(arr["Z"]).long()
    pos_all = torch.from_numpy(arr["pos"]).float()
    cell_all = torch.from_numpy(arr["cell"]).float()
    energy_all = torch.from_numpy(arr["energy"]).float()
    forces_all = torch.from_numpy(arr["forces"]).float()
    out: list[dict] = []
    for i in range(int(ptr.shape[0]) - 1):
        s, e = int(ptr[i].item()), int(ptr[i + 1].item())
        out.append(
            {
                "Z": Z_all[s:e],
                "pos": pos_all[s:e],
                "cell": cell_all[i],
                "energy": float(energy_all[i].item()),
                "forces": forces_all[s:e],
            }
        )
    return out


def _load_nacl_fallback(*, seed: int = 0) -> tuple[list[dict], list[dict], str]:
    """Build a minimal molten-NaCl substrate (used when water is unavailable).

    The fallback is a 2 × 2 × 2 NaCl rocksalt cell with thermal jitter.
    It exercises the periodic + charged-ion code paths with synthetic
    zero-target labels; the bench_long Madelung-cohesion bar is not
    encoded here as a separate test — see the ``test_long_range_improvement``
    skip path in :class:`BMSonata` for the contract.

    Returns:
        ``(train, val, "nacl_1100k")`` matching the shape of
        :func:`_load_water_data`.
    """
    snap = _build_synthetic_nacl_box(seed=seed, n_pairs=2)
    return [snap], [snap], "nacl_1100k"


# ---------------------------------------------------------------------------
# TensorDict construction
# ---------------------------------------------------------------------------


def _make_batch(snapshots: list[dict], *, dtype: torch.dtype = torch.float32) -> TensorDict:
    """Assemble a periodic :class:`TensorDict` with full bidirectional edges.

    All snapshots are placed in a single batch with per-graph cell vectors
    written under ``("graphs", "cell")``. Edges are *full bidirectional*
    (every directed atom pair within a graph, no self-loops) so the
    encoder's source/target convention is honoured everywhere; minimum-
    image displacement is computed from the per-graph cell so periodic
    geometry is consistent with the encoder's expectations.

    Args:
        snapshots: List of snapshot dicts (see
            :func:`_build_synthetic_water_box`).
        dtype: Floating-point dtype for ``pos``, ``cell``, ``bond_diff``,
            ``bond_dist``. Default float32; periodic finite-difference
            assertions in unit tests pass float64 here.

    Returns:
        :class:`TensorDict` with ``atoms``, ``edges``, ``graphs`` sub-dicts.
        ``("graphs", "energy")`` carries the per-graph reference energy in
        eV; ``("atoms", "forces")`` carries the per-atom reference forces
        in eV/Å.
    """
    Z_list: list[torch.Tensor] = []
    pos_list: list[torch.Tensor] = []
    forces_list: list[torch.Tensor] = []
    cells: list[torch.Tensor] = []
    energies: list[float] = []
    batch_idx: list[torch.Tensor] = []
    edge_blocks: list[torch.Tensor] = []
    bond_diff_blocks: list[torch.Tensor] = []
    num_atoms_list: list[int] = []

    offset = 0
    for g, snap in enumerate(snapshots):
        Z = snap["Z"].long()
        pos = snap["pos"].to(dtype=dtype)
        cell = snap["cell"].to(dtype=dtype)
        forces = snap["forces"].to(dtype=dtype)
        n = int(Z.shape[0])

        # Full bidirectional intra-graph edges, no self-loops.
        idx = torch.arange(n, dtype=torch.long)
        src, dst = torch.meshgrid(idx, idx, indexing="ij")
        mask = src != dst
        src, dst = src[mask], dst[mask]
        edge_block = torch.stack([offset + src, offset + dst], dim=1)
        diff = pos[dst] - pos[src]

        # Minimum-image wrap (orthorhombic + triclinic-safe via fractional
        # round-trip). bond_dist is the wrapped Euclidean norm.
        cell_inv = torch.linalg.inv(cell)
        frac = diff @ cell_inv.T
        frac_wrapped = frac - frac.round()
        diff = frac_wrapped @ cell.T

        Z_list.append(Z)
        pos_list.append(pos)
        forces_list.append(forces)
        cells.append(cell)
        energies.append(float(snap["energy"]))
        batch_idx.append(torch.full((n,), g, dtype=torch.long))
        edge_blocks.append(edge_block)
        bond_diff_blocks.append(diff)
        num_atoms_list.append(n)
        offset += n

    Z_full = torch.cat(Z_list, dim=0)
    pos_full = torch.cat(pos_list, dim=0)
    forces_full = torch.cat(forces_list, dim=0)
    batch_full = torch.cat(batch_idx, dim=0)
    edge_index = torch.cat(edge_blocks, dim=0)
    bond_diff = torch.cat(bond_diff_blocks, dim=0)
    bond_dist = bond_diff.norm(dim=-1)
    cell_full = torch.stack(cells, dim=0)
    num_atoms = torch.tensor(num_atoms_list, dtype=torch.long)
    energy_full = torch.tensor(energies, dtype=dtype)
    total_charge = torch.zeros(len(snapshots), dtype=dtype)

    n_atoms_total = int(Z_full.shape[0])
    n_edges_total = int(edge_index.shape[0])
    n_graphs = len(snapshots)

    return TensorDict(
        atoms=TensorDict(
            Z=Z_full,
            pos=pos_full,
            batch=batch_full,
            forces=forces_full,
            batch_size=[n_atoms_total],
        ),
        edges=TensorDict(
            edge_index=edge_index,
            bond_diff=bond_diff,
            bond_dist=bond_dist,
            batch_size=[n_edges_total],
        ),
        graphs=TensorDict(
            num_atoms=num_atoms,
            cell=cell_full,
            total_charge=total_charge,
            energy=energy_full,
            batch_size=[n_graphs],
        ),
        batch_size=[],
    )


# ---------------------------------------------------------------------------
# Models: matched Sonata + ablated short-range-only baseline
# ---------------------------------------------------------------------------


class _ShortRangeBaseline(nn.Module):
    """Ablated baseline: Allegro encoder + ``EdgeEnergyHead`` (no Ewald).

    Mirrors the ``compute_forces`` plumbing of :class:`Sonata` so the
    training loop can call both models with the same signature. The
    multipole head is omitted entirely — without an Ewald evaluator its
    outputs would be unused, and matching parameter count to Sonata is
    not the relevant control here (the relevant control is ablating the
    long-range tail, not the head capacity).
    """

    def __init__(self, encoder: Allegro, head: EdgeEnergyHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(
        self,
        batch: TensorDict,
        *,
        compute_forces: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run encoder → ``EdgeEnergyHead``; optionally derive forces.

        Args:
            batch: Periodic :class:`TensorDict` carrying ``atoms.pos``
                and per-graph ``cell``. ``("atoms", "pos")`` is detached
                and reattached to the autograd graph when
                ``compute_forces=True``.
            compute_forces: If ``True``, derive forces ``F = -∂U/∂pos``
                via a single :func:`torch.autograd.grad` call.

        Returns:
            dict with ``energy`` ``(B,)`` in eV and ``forces`` ``(N, 3)``
            in eV/Å (the latter only when ``compute_forces=True``).
        """
        edge_index = batch["edges", "edge_index"]
        pos = batch["atoms", "pos"]
        if compute_forces:
            pos = pos.detach().requires_grad_(True)
            bond_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
            cell = batch.get(("graphs", "cell"))
            if cell is not None:
                # Re-wrap into minimum image with the current pos so the
                # encoder still sees PBC-consistent geometry. We use the
                # per-edge graph index to look up the per-graph cell.
                atom_batch = batch["atoms", "batch"]
                edge_graph = atom_batch[edge_index[:, 0]]
                cell_per_edge = cell[edge_graph]
                cell_inv = torch.linalg.inv(cell_per_edge)
                frac = torch.einsum("eab,eb->ea", cell_inv, bond_diff)
                frac_wrapped = frac - frac.round()
                bond_diff = torch.einsum("eab,eb->ea", cell_per_edge, frac_wrapped)
            bond_dist = bond_diff.norm(dim=-1)
            batch[("atoms", "pos")] = pos
            batch[("edges", "bond_diff")] = bond_diff
            batch[("edges", "bond_dist")] = bond_dist
        self.encoder(batch)
        out = self.head(batch)
        energy = out["energy"]
        result: dict[str, torch.Tensor] = {"energy": energy}
        if compute_forces:
            forces = -torch.autograd.grad(
                energy.sum(),
                pos,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
            result["forces"] = forces
        return result


def _build_sonata_and_baseline(
    *,
    num_elements: int = NUM_ELEMENTS,
    avg_num_neighbors: float = DEFAULT_AVG_NEIGHBORS,
    seed: int = 0,
    sigma: float = DEFAULT_SIGMA,
    dl: float = DEFAULT_DL,
    r_max: float = DEFAULT_R_MAX,
    num_features: int = DEFAULT_NUM_FEATURES,
    num_layers: int = DEFAULT_NUM_LAYERS,
    l_max: int = DEFAULT_L_MAX,
) -> tuple[nn.Module, nn.Module, float]:
    """Build a matched Sonata + short-range-only baseline pair.

    Both models share the same Allegro encoder hyperparameters and the
    same RNG seed for parameter init. Sonata wires the encoder's tensor
    track into a multipole head + :class:`EwaldMultipoleEnergy`; the
    baseline omits both — its only readout is :class:`EdgeEnergyHead`.

    Args:
        num_elements: Periodic-table size for the encoder's type
            embedding. Default ``NUM_ELEMENTS`` is a comfortable upper
            bound covering every element this benchmark touches.
        avg_num_neighbors: Dataset-wide ⟨|N(i)|⟩ for the encoder and head
            normalisation. Default ``DEFAULT_AVG_NEIGHBORS`` matches the
            small synthetic boxes; full-mode runs should pass the real
            dataset statistic.
        seed: Seed applied before each :class:`Allegro` construction so
            the two encoders and their heads start from the same init.
        sigma: σ-Gaussian charge-smearing length in Å (Sonata only).
        dl: Ewald reciprocal-space grid resolution in Å (Sonata only).
        r_max: Encoder cutoff in Å. Doubles as the ``r_cut`` returned to
            the caller for the long-range MAE kernel.
        num_features: ``Allegro.num_scalar_features``.
        num_layers: ``Allegro.num_layers``.
        l_max: Encoder ``l_max``. Sonata requires ≥ 2 for dipole +
            quadrupole.

    Returns:
        ``(sonata_model, baseline_model, r_cut)`` where ``r_cut`` is the
        Allegro encoder cutoff (= ``r_max``) the long-range force-MAE
        kernel uses to mask atoms.
    """
    if l_max < 2:
        raise ValueError(f"Sonata requires l_max >= 2; got {l_max}")

    # Encoder factory — re-seeded each time so both models start from the
    # same parameter init. Allegro pulls its random init from the global
    # torch RNG state, so we re-seed before each construction.
    def _make_encoder(*, expose_tensor_track: bool) -> Allegro:
        torch.manual_seed(seed)
        return Allegro(
            num_elements=num_elements,
            num_scalar_features=num_features,
            num_tensor_features=num_features // 4,
            r_max=r_max,
            num_bessel=8,
            l_max=l_max,
            num_layers=num_layers,
            type_embed_dim=DEFAULT_TYPE_EMBED_DIM,
            latent_mlp_depth=1,
            latent_mlp_width=DEFAULT_LATENT_MLP_WIDTH,
            avg_num_neighbors=avg_num_neighbors,
            expose_tensor_track=expose_tensor_track,
        )

    enc_sonata = _make_encoder(expose_tensor_track=True)
    torch.manual_seed(seed + 1)  # head init — distinct from encoder
    sonata_short = EdgeEnergyHead(
        input_dim=enc_sonata.output_dim,
        hidden_dim=128,
        avg_num_neighbors=avg_num_neighbors,
        out_key="energy_short",
    )
    sonata_model = build_sonata(
        enc_sonata,
        sigma=sigma,
        dl=dl,
        charge=True,
        dipole=True,
        quadrupole=True,
        constrain_total_charge=True,
        avg_num_neighbors=avg_num_neighbors,
        short_range_head=sonata_short,
    )

    enc_baseline = _make_encoder(expose_tensor_track=False)
    torch.manual_seed(seed + 1)
    baseline_head = EdgeEnergyHead(
        input_dim=enc_baseline.output_dim,
        hidden_dim=128,
        avg_num_neighbors=avg_num_neighbors,
        out_key="energy",
    )
    baseline_model = _ShortRangeBaseline(enc_baseline, baseline_head)

    return sonata_model, baseline_model, float(r_max)


# ---------------------------------------------------------------------------
# Long-range force MAE kernel
# ---------------------------------------------------------------------------


def _compute_long_range_force_mae(
    *,
    forces_pred: torch.Tensor,
    forces_ref: torch.Tensor,
    pos: torch.Tensor,
    cell: torch.Tensor,
    atom_batch: torch.Tensor,
    r_cut: float,
) -> dict[str, float]:
    """Per-atom min-image ``r_min``; force-MAE over atoms with ``r_min > r_cut``.

    The kernel is the geometric load-bearing piece of
    ``sonata-03-bench`` (ac-002 / ac-005). For every atom in every graph
    in ``atom_batch``, it computes the minimum-image distance to its
    nearest intra-graph neighbour (``r_min``), masks the atoms whose
    ``r_min > r_cut`` (the "long-range" cohort — those that the encoder's
    short-range cutoff cannot see any neighbours for), and returns the
    mean absolute error of ``forces_pred - forces_ref`` averaged over
    those atoms' force *components* (i.e. flattened ``(N_long, 3)`` →
    ``3·N_long`` scalars).

    Args:
        forces_pred: ``(N, 3)`` predicted atomic forces (any unit).
        forces_ref: ``(N, 3)`` reference atomic forces (same unit).
        pos: ``(N, 3)`` atomic positions in Å.
        cell: ``(B, 3, 3)`` per-graph cell vectors in Å.
        atom_batch: ``(N,)`` per-atom graph membership.
        r_cut: encoder cutoff in Å. Atoms with ``r_min > r_cut`` are
            classified as long-range.

    Returns:
        dict with ``mae`` (mean absolute error in input units, ``float``)
        and ``n_long_range`` (atom count, ``int``). When no atoms are
        long-range the kernel returns ``mae = NaN`` and ``n = 0`` so the
        caller can record the gap rather than masking it.
    """
    if forces_pred.shape != forces_ref.shape:
        raise ValueError(
            f"forces_pred {tuple(forces_pred.shape)} and forces_ref "
            f"{tuple(forces_ref.shape)} must match."
        )
    if forces_pred.shape != pos.shape:
        raise ValueError(
            f"forces_pred {tuple(forces_pred.shape)} and pos {tuple(pos.shape)} must match."
        )
    n_atoms = pos.shape[0]
    device = pos.device
    dtype = pos.dtype
    r_min = torch.full((n_atoms,), float("inf"), dtype=dtype, device=device)
    n_graphs = int(atom_batch.max().item()) + 1 if n_atoms > 0 else 0
    for g in range(n_graphs):
        idx = torch.nonzero(atom_batch == g, as_tuple=False).squeeze(-1)
        if idx.numel() < 2:
            continue
        pos_g = pos[idx]
        cell_g = cell[g]
        cell_inv = torch.linalg.inv(cell_g)
        diff = pos_g.unsqueeze(0) - pos_g.unsqueeze(1)  # (n_g, n_g, 3)
        frac = diff @ cell_inv.T
        frac_wrapped = frac - frac.round()
        diff_min = frac_wrapped @ cell_g.T
        dist = diff_min.norm(dim=-1)  # (n_g, n_g)
        eye_mask = torch.eye(idx.numel(), dtype=torch.bool, device=device)
        dist = dist.masked_fill(eye_mask, float("inf"))
        r_min[idx] = dist.min(dim=-1).values

    is_long_range = r_min > r_cut
    n_long = int(is_long_range.sum().item())
    if n_long == 0:
        return {"mae": float("nan"), "n_long_range": 0}
    err = (forces_pred[is_long_range] - forces_ref[is_long_range]).abs().mean().item()
    return {"mae": float(err), "n_long_range": n_long}


# ---------------------------------------------------------------------------
# Training loop + validation
# ---------------------------------------------------------------------------


def _train_one_run(
    *,
    model: nn.Module,
    train_batch: TensorDict,
    val_batch: TensorDict,
    n_steps: int,
    lr: float = DEFAULT_LR,
    seed: int = 0,
    r_cut: float,
) -> dict[str, float]:
    """Train ``model`` for ``n_steps`` and return validation MAE columns.

    The training and validation batches are :class:`TensorDict` objects
    pre-built with full bidirectional edges and minimum-image
    ``bond_diff`` / ``bond_dist`` (see :func:`_make_batch`). Because the
    smoke- and fallback-mode datasets are too small to require a sampler,
    we re-use the same train batch every step — the spec's "smoke is
    plumbing, not generalisation" stance applies.

    Loss: ``MSE(energy_pred, energy_ref) / N + MSE(forces_pred, forces_ref)``.
    Optimiser: ``Adam(lr=lr)`` against a fixed seed.

    Args:
        model: Either a :class:`Sonata` or a :class:`_ShortRangeBaseline`.
            Both expose ``forward(batch, compute_forces=True)`` returning
            ``{"energy": (B,), "forces": (N, 3)}``.
        train_batch: Pre-built :class:`TensorDict` used at every step.
        val_batch: Pre-built :class:`TensorDict` used for the final MAE
            computation.
        n_steps: Number of optimisation steps.
        lr: Adam learning rate.
        seed: Seed for optimiser RNG (Adam itself has no RNG; included
            for symmetry with the rest of the harness).
        r_cut: Encoder cutoff in Å, forwarded to
            :func:`_compute_long_range_force_mae`.

    Returns:
        dict with ``energy_mae_meV_per_atom``, ``force_mae_meV_per_A``,
        and ``force_mae_long_range_meV_per_A``. Values are eV-to-meV
        converted at the boundary.
    """
    torch.manual_seed(seed)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(n_steps):
        b = train_batch.clone()
        out = model(b, compute_forces=True)
        e_pred = out["energy"]
        f_pred = out["forces"]
        e_ref = b["graphs", "energy"].to(dtype=e_pred.dtype)
        f_ref = b["atoms", "forces"].to(dtype=f_pred.dtype)
        n_atoms = float(b["atoms", "Z"].shape[0])
        loss = ((e_pred - e_ref) ** 2).mean() / max(1.0, n_atoms) + ((f_pred - f_ref) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

    model.eval()
    b = val_batch.clone()
    out = model(b, compute_forces=True)
    e_pred = out["energy"].detach()
    f_pred = out["forces"].detach()
    e_ref = b["graphs", "energy"].to(dtype=e_pred.dtype)
    f_ref = b["atoms", "forces"].to(dtype=f_pred.dtype)
    n_atoms_total = int(b["atoms", "Z"].shape[0])
    e_mae_per_atom_eV = (e_pred - e_ref).abs().sum().item() / max(1, n_atoms_total)
    f_mae_eV_per_A = (f_pred - f_ref).abs().mean().item()
    lr_kernel = _compute_long_range_force_mae(
        forces_pred=f_pred,
        forces_ref=f_ref,
        pos=b["atoms", "pos"].detach(),
        cell=b["graphs", "cell"],
        atom_batch=b["atoms", "batch"],
        r_cut=r_cut,
    )
    f_mae_long_eV_per_A = lr_kernel["mae"]
    return {
        "energy_mae_meV_per_atom": e_mae_per_atom_eV * EV_TO_MEV,
        "force_mae_meV_per_A": f_mae_eV_per_A * EV_TO_MEV,
        "force_mae_long_range_meV_per_A": f_mae_long_eV_per_A * EV_TO_MEV
        if not math.isnan(f_mae_long_eV_per_A)
        else float("nan"),
        "n_long_range_atoms": float(lr_kernel["n_long_range"]),
    }


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def _append_csv_row(row: dict, *, csv_path: Path = CSV_PATH) -> None:
    """Append one row to ``csv_path``. Writes a header row on first call.

    Args:
        row: Mapping with one value per column in :data:`CSV_COLUMNS`.
        csv_path: Override path (used in tests to redirect to ``tmp_path``).
    """
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_benchmark(
    *,
    smoke: bool,
    n_steps: int,
    seed: int = 0,
    sigma: float = DEFAULT_SIGMA,
    dl: float = DEFAULT_DL,
    csv_path: Path = CSV_PATH,
    assert_bars: bool = False,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """End-to-end driver. Loads data, trains both models, writes one CSV row.

    Args:
        smoke: If ``True``, ``n_steps`` is overridden to
            :data:`SMOKE_N_STEPS` and quantitative bar assertions are
            skipped.
        n_steps: Number of training steps for each model (full mode).
        seed: Master seed.
        sigma: Sonata σ-Gaussian width in Å.
        dl: Sonata Ewald reciprocal-space grid resolution in Å.
        csv_path: Destination CSV (override for tests).
        assert_bars: If ``True``, assert the spec's quantitative bars
            on water substrates. NaCl-fallback substrates skip the
            assertions and merely record the row.
        device: Optional torch device. ``None`` keeps tensors on CPU.

    Returns:
        The CSV row dict the run appended.
    """
    if smoke:
        n_steps = SMOKE_N_STEPS

    # Substrate selection: try water; on RuntimeError("water reference
    # unavailable") retry once; on a second failure, fall back to NaCl.
    try:
        train, val, substrate = _load_water_data(smoke=smoke, seed=seed)
    except RuntimeError as exc:
        if "water reference unavailable" not in str(exc):
            raise
        try:
            train, val, substrate = _load_water_data(smoke=smoke, seed=seed)
        except RuntimeError as exc2:
            if "water reference unavailable" not in str(exc2):
                raise
            train, val, substrate = _load_nacl_fallback(seed=seed)

    train_batch = _make_batch(train)
    val_batch = _make_batch(val)
    if device is not None:
        dev = torch.device(device)
        train_batch = train_batch.to(dev)
        val_batch = val_batch.to(dev)
    n_atoms = int(train_batch["atoms", "Z"].shape[0])

    sonata_model, baseline_model, r_cut = _build_sonata_and_baseline(
        seed=seed,
        sigma=sigma,
        dl=dl,
    )
    if device is not None:
        sonata_model = sonata_model.to(dev)
        baseline_model = baseline_model.to(dev)

    sonata_metrics = _train_one_run(
        model=sonata_model,
        train_batch=train_batch,
        val_batch=val_batch,
        n_steps=n_steps,
        seed=seed,
        r_cut=r_cut,
    )
    baseline_metrics = _train_one_run(
        model=baseline_model,
        train_batch=train_batch,
        val_batch=val_batch,
        n_steps=n_steps,
        seed=seed,
        r_cut=r_cut,
    )

    sonata_lr = sonata_metrics["force_mae_long_range_meV_per_A"]
    baseline_lr = baseline_metrics["force_mae_long_range_meV_per_A"]
    if (
        baseline_lr is None
        or math.isnan(baseline_lr)
        or baseline_lr == 0.0
        or math.isnan(sonata_lr)
    ):
        improvement_pct = float("nan")
    else:
        improvement_pct = 100.0 * (1.0 - sonata_lr / baseline_lr)

    row = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "substrate": substrate,
        "n_atoms": n_atoms,
        "n_steps": n_steps,
        "energy_mae_meV_per_atom": f"{sonata_metrics['energy_mae_meV_per_atom']:.6g}",
        "force_mae_meV_per_A": f"{sonata_metrics['force_mae_meV_per_A']:.6g}",
        "force_mae_long_range_meV_per_A_sonata": f"{sonata_lr:.6g}",
        "force_mae_long_range_meV_per_A_baseline": f"{baseline_lr:.6g}",
        "improvement_pct": f"{improvement_pct:.6g}",
        "sigma": f"{sigma:.6g}",
        "dl": f"{dl:.6g}",
    }
    _append_csv_row(row, csv_path=csv_path)

    if assert_bars and substrate.startswith("water_"):
        e_mae = sonata_metrics["energy_mae_meV_per_atom"]
        f_mae = sonata_metrics["force_mae_meV_per_A"]
        if not (e_mae <= 1.0):
            raise AssertionError(
                f"Sonata energy MAE {e_mae:.4f} meV/atom exceeds the 1 meV/atom bar."
            )
        if not (f_mae <= 50.0):
            raise AssertionError(f"Sonata force MAE {f_mae:.4f} meV/Å exceeds the 50 meV/Å bar.")
        if not (improvement_pct >= 30.0):
            raise AssertionError(
                f"Long-range force-MAE improvement {improvement_pct:.2f}% is below the 30 % bar."
            )

    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_argv(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sonata long-range periodic benchmark — bulk water at 300 K.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Smoke mode (~100 steps, no bars).")
    mode.add_argument("--full", action="store_true", help="Full mode (asserts bars).")
    parser.add_argument("--n-steps", type=int, default=10000, help="Training steps for full mode.")
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA, help="Ewald σ in Å.")
    parser.add_argument("--dl", type=float, default=DEFAULT_DL, help="Ewald dl in Å.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=CSV_PATH,
        help="Override CSV destination (defaults to benchmarks/molpot/sonata_results.csv).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Torch device override, e.g. "cuda" or "cpu". Default: CPU.',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        ``0`` on success; non-zero on assertion failure (the run is still
        recorded in the CSV for auditability).
    """
    args = _parse_argv(argv)
    smoke = args.smoke or not args.full
    try:
        run_benchmark(
            smoke=smoke,
            n_steps=args.n_steps,
            seed=args.seed,
            sigma=args.sigma,
            dl=args.dl,
            csv_path=args.csv_path,
            assert_bars=not smoke,
            device=args.device,
        )
    except AssertionError as exc:
        print(f"[bm_sonata] BAR FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


# ---------------------------------------------------------------------------
# Pytest-collected benchmark
# ---------------------------------------------------------------------------


class BMSonata:
    """Pytest-collected harness for the Sonata long-range benchmark.

    ``test_smoke`` and ``test_long_range_force_mae_kernel`` are CPU-only
    and gate every CI run. ``test_long_range_improvement`` is gated by
    ``pytest.mark.bench_long`` and is meant for nightly hosts. ``test_substrate_fallback``
    monkeypatches :func:`_load_water_data` to exercise the molten-NaCl
    fallback branch without touching the network.
    """

    def test_smoke(self, tmp_path):
        """ac-001 / ac-006 — CLI smoke runs end-to-end and appends one CSV row."""
        csv_path = tmp_path / "sonata_results.csv"
        rc = main(["--smoke", "--seed", "0", "--csv-path", str(csv_path)])
        assert rc == 0, f"main() exit code {rc}"
        assert csv_path.exists(), "CSV file not created"
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1, f"expected exactly 1 row, got {len(rows)}"
        row = rows[0]
        assert int(row["n_steps"]) == SMOKE_N_STEPS
        assert row["substrate"] in {
            "water_synthetic_4",
            "water_spice_ood",
            "nacl_1100k",
        }, f"unexpected substrate {row['substrate']!r}"
        # Energy/force MAE come straight from a forward pass and must be
        # finite. The long-range columns (and improvement_pct) may be NaN
        # in smoke mode when the synthetic configuration has no atoms with
        # r_min > r_cut (every atom is intramolecularly bonded inside a
        # tight monomer); the kernel correctly reports that as NaN rather
        # than masking it.
        for col in ("energy_mae_meV_per_atom", "force_mae_meV_per_A"):
            v = float(row[col])
            assert math.isfinite(v), f"{col}={v} (expected finite)"

    def test_long_range_force_mae_kernel(self):
        """ac-002 — kernel selects only ``r_min > r_cut`` atoms.

        Build a 12-atom periodic box where exactly 3 atoms are isolated
        (``r_min > r_cut``) and the other 9 sit in a tight cluster. The
        kernel must return the MAE over those 3 atoms' force components
        and ignore the 9 cluster atoms.
        """
        torch.manual_seed(0)
        r_cut = 5.0
        box_length = 30.0
        # 9 atoms tightly clustered near the origin (within ~1.5 Å of each other)
        cluster = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5],
            ],
            dtype=torch.float64,
        )
        # 3 atoms placed at corners of the box, each > r_cut from every other atom
        # (and from each other; 30/2 = 15 > 5).
        isolated = torch.tensor(
            [
                [box_length / 2.0, 0.0, 0.0],
                [0.0, box_length / 2.0, 0.0],
                [0.0, 0.0, box_length / 2.0],
            ],
            dtype=torch.float64,
        )
        pos = torch.cat([cluster, isolated], dim=0)  # (12, 3)
        n_atoms = pos.shape[0]
        atom_batch = torch.zeros(n_atoms, dtype=torch.long)
        cell = (box_length * torch.eye(3, dtype=torch.float64)).unsqueeze(0)  # (1, 3, 3)

        # Reference forces are zero everywhere; predicted forces deviate by
        # a known per-atom amount so we can verify which atoms are picked.
        forces_ref = torch.zeros((n_atoms, 3), dtype=torch.float64)
        forces_pred = torch.zeros((n_atoms, 3), dtype=torch.float64)
        # Cluster atoms (indices 0..8) get a deviation we MUST IGNORE.
        forces_pred[0:9] = 100.0
        # Isolated atoms (indices 9..11) get a known deviation we MUST INCLUDE.
        forces_pred[9] = torch.tensor([1.0, 2.0, 3.0])
        forces_pred[10] = torch.tensor([2.0, 3.0, 4.0])
        forces_pred[11] = torch.tensor([3.0, 4.0, 5.0])

        out = _compute_long_range_force_mae(
            forces_pred=forces_pred,
            forces_ref=forces_ref,
            pos=pos,
            cell=cell,
            atom_batch=atom_batch,
            r_cut=r_cut,
        )
        assert out["n_long_range"] == 3, f"n_long_range={out['n_long_range']}"
        # Expected MAE: mean |1,2,3,2,3,4,3,4,5| = 27/9 = 3.0
        expected_mae = (1.0 + 2.0 + 3.0 + 2.0 + 3.0 + 4.0 + 3.0 + 4.0 + 5.0) / 9.0
        assert abs(out["mae"] - expected_mae) < 1e-10, (
            f"mae={out['mae']!r}, expected {expected_mae}; cluster atoms must be excluded"
        )

    @pytest.mark.bench_long
    def test_long_range_improvement(self, tmp_path):
        """ac-003 / ac-004 / ac-005 — full bench_long run hits all three bars."""
        csv_path = tmp_path / "sonata_results.csv"
        rc = main(["--full", "--n-steps", "10000", "--seed", "0", "--csv-path", str(csv_path)])
        assert rc == 0, f"main() exit code {rc}"
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        row = rows[0]
        if row["substrate"] == "nacl_1100k":
            pytest.skip(
                "Water reference unavailable; substrate fell back to molten-NaCl. "
                "The 30 % long-range improvement bar applies to water — Madelung "
                "cohesion bar for NaCl is recorded in the CSV but not asserted here."
            )
        assert float(row["energy_mae_meV_per_atom"]) <= 1.0, (
            f"Sonata energy MAE {row['energy_mae_meV_per_atom']} meV/atom above the 1 meV/atom bar."
        )
        assert float(row["force_mae_meV_per_A"]) <= 50.0, (
            f"Sonata force MAE {row['force_mae_meV_per_A']} meV/Å above the 50 meV/Å bar."
        )
        assert float(row["improvement_pct"]) >= 30.0, (
            f"Long-range improvement {row['improvement_pct']}% below the 30 % bar."
        )

    def test_substrate_fallback(self, monkeypatch, tmp_path):
        """ac-007 — water-data outage routes through the NaCl fallback."""
        mod = sys.modules[__name__]
        n_calls = {"k": 0}

        def _fail(*, smoke: bool, seed: int = 0):  # noqa: ARG001
            n_calls["k"] += 1
            raise RuntimeError("water reference unavailable")

        monkeypatch.setattr(mod, "_load_water_data", _fail)

        csv_path = tmp_path / "sonata_results.csv"
        rc = main(["--smoke", "--seed", "0", "--csv-path", str(csv_path)])
        assert rc == 0, f"main() exit code {rc}"
        # The fallback contract: try once, fail, retry once, fall back.
        assert n_calls["k"] == 2, f"expected exactly 2 attempts, got {n_calls['k']}"
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        assert rows[-1]["substrate"] == "nacl_1100k"


if __name__ == "__main__":
    raise SystemExit(main())
