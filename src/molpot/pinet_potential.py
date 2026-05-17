"""PiNet energy and force prediction model.

Combines a :class:`~molzoo.PiNet` encoder with per-atom energy readout,
graph-level energy aggregation, and autograd force derivation.

Reference:
    Li et al. "PiNN: Equivariant Neural Network Suite for Modeling
    Electrochemical Systems", JCTC 2025.
    https://doi.org/10.1021/acs.jctc.4c01570
"""

from __future__ import annotations

from typing import Literal, Mapping

import torch
import torch.nn as nn

from molix import config
from molix.data.types import GraphBatch
from molpot.derivation import EnergyAggregation, ForceDerivation


def _recompute_edges(batch: GraphBatch) -> None:
    """Recompute edge geometry from positions so autograd traces forces."""
    pos = batch["atoms", "pos"]
    ei = batch["edges", "edge_index"]
    diff = pos[ei[:, 1]] - pos[ei[:, 0]]
    batch["edges", "bond_diff"] = diff
    batch["edges", "bond_dist"] = diff.norm(dim=-1).clamp(min=1e-8)


def _pool_layer(features: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return features.mean(dim=1)
    if reduction == "sum":
        return features.sum(dim=1)
    if reduction == "last":
        return features[:, -1]
    raise ValueError(f"Unknown reduction {reduction!r}.")


class PiNetPotential(nn.Module):
    """PiNet energy + force prediction model.

    Args:
        encoder: :class:`~molzoo.PiNet` encoder (or any module that writes
            ``("atoms", "node_features")`` into a ``GraphBatch``).
        hidden_dim: Hidden dimension of the per-atom energy MLP.
        layer_reduction: How to pool across GC-block layers
            (``"mean"``, ``"sum"``, or ``"last"``).
        e_dress: Optional per-element energy corrections ``{Z: eV}``.
        e_scale: Divisor applied to total energy (e.g. for unit conversion).
        e_unit: Multiplier applied to total energy.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        hidden_dim: int = 64,
        layer_reduction: Literal["mean", "sum", "last"] = "mean",
        e_dress: dict[int, float] | None = None,
        e_scale: float = 1.0,
        e_unit: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.layer_reduction = layer_reduction
        self.e_dress = e_dress or {}
        self.e_scale = e_scale
        self.e_unit = e_unit

        input_dim: int = getattr(encoder, "output_dim", 16)
        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=config.ftype),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, dtype=config.ftype),
        )
        self.energy_aggregation = EnergyAggregation(pooling="sum")
        self.force_derivation = ForceDerivation()

    def forward(
        self, batch: GraphBatch, *, compute_forces: bool = False
    ) -> dict[str, torch.Tensor]:
        if compute_forces and not batch["atoms", "pos"].requires_grad:
            batch["atoms", "pos"] = batch["atoms", "pos"].clone().requires_grad_(True)
        _recompute_edges(batch)
        batch = self.encoder(batch)

        node_feats = _pool_layer(batch["atoms", "node_features"], self.layer_reduction)
        atom_energy = self.node_mlp(node_feats).squeeze(-1)

        atom_batch = batch["atoms", "batch"]
        num_graphs = int(atom_batch.max().item()) + 1 if atom_batch.numel() else 0
        energy = self.energy_aggregation(atom_energy, atom_batch, num_graphs=num_graphs)

        if self.e_dress:
            energy = energy + _atomic_dress(
                batch["atoms", "Z"], atom_batch, self.e_dress, num_graphs,
            ).to(dtype=energy.dtype)
        energy = (energy / self.e_scale) * self.e_unit

        out: dict[str, torch.Tensor] = {"atomic_energy": atom_energy, "energy": energy}
        if compute_forces:
            out["forces"] = self.force_derivation(energy, batch["atoms", "pos"])
        return out


def _atomic_dress(
    Z: torch.Tensor, batch: torch.Tensor, dress: Mapping[int, float], num_graphs: int,
) -> torch.Tensor:
    values = torch.zeros_like(Z, dtype=torch.float32)
    for z_val, e_val in dress.items():
        values = torch.where(Z == int(z_val), torch.full_like(values, float(e_val)), values)
    out = torch.zeros(num_graphs, dtype=values.dtype, device=values.device)
    out.scatter_add_(0, batch, values)
    return out
