"""PiNet dipole prediction model.

Supports PiNN-style dipole variants: atomic charge (``ac``), atomic dipole
(``ad``), bond-charge (``bc``), oxidation-state (``os``), and combinations.

Reference:
    Li et al. "PiNN: Equivariant Neural Network Suite for Modeling
    Electrochemical Systems", JCTC 2025.
    https://doi.org/10.1021/acs.jctc.4c01570
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from molix import config
from molix.data.types import GraphBatch


def _pool_layer(features: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return features.mean(dim=1)
    if reduction == "sum":
        return features.sum(dim=1)
    if reduction == "last":
        return features[:, -1]
    raise ValueError(f"Unknown reduction {reduction!r}.")


def _num_graphs(batch: torch.Tensor) -> int:
    return int(batch.max().item()) + 1 if batch.numel() else 0


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
    if src.numel() == 0:
        return out
    expand_index = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add_(0, expand_index, src)


def _recompute_edges(batch: GraphBatch) -> None:
    pos = batch["atoms", "pos"]
    ei = batch["edges", "edge_index"]
    diff = pos[ei[:, 1]] - pos[ei[:, 0]]
    batch["edges", "bond_diff"] = diff
    batch["edges", "bond_dist"] = diff.norm(dim=-1).clamp(min=1e-8)


def _graph_counts(batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    ones = torch.ones(batch.shape[0], dtype=config.ftype, device=batch.device)
    return _scatter_sum(ones, batch, num_graphs).clamp(min=1.0)


def _charge_neutralize(
    charges: torch.Tensor,
    batch: torch.Tensor,
    *,
    total_charge: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num = _num_graphs(batch)
    pre = _scatter_sum(charges, batch, num)
    if total_charge is None:
        total_charge = torch.zeros_like(pre)
    correction = (pre - total_charge.view_as(pre)) / _graph_counts(batch, num)
    charges = charges - correction[batch]
    post = _scatter_sum(charges, batch, num)
    return charges, pre, post


class PiNetDipole(nn.Module):
    """PiNet molecular dipole prediction model.

    Variants are composed from PiNN model families:

    * ``"ac"``: atomic charge dipole ``Σ_i q_i r_i``.
    * ``"ad"``: atomic dipole ``Σ_i μ_i`` from P3 features.
    * ``"bc"``: bond-charge dipole ``Σ_{ij} q_{ij} r_{ij}``.
    * ``"ad_os"``: atomic dipole + oxidation-state charge term.
    * Combinations: ``"ac_ad"``, ``"ac_bc"``, ``"ad_bc"``.

    Args:
        encoder: :class:`~molzoo.PiNet` encoder.
        hidden_dim: Hidden dimension for charge/dipole MLPs.
        variant: Dipole variant string.
        layer_reduction: How to pool across GC-block layers.
        vector_dipole: If True, output ``"dipole"`` as a vector ``(B, 3)``;
            otherwise output norm.
        charge_neutrality: Apply charge neutralization to AC predictions.
        regularization: Add ``"bond_charge_l2"`` penalty for BC variants.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        hidden_dim: int = 64,
        variant: str = "ac_ad",
        layer_reduction: Literal["mean", "sum", "last"] = "mean",
        vector_dipole: bool = True,
        charge_neutrality: bool = True,
        regularization: bool = True,
    ) -> None:
        super().__init__()
        variant = variant.lower()
        self.encoder = encoder
        self.variant = variant
        self.layer_reduction = layer_reduction
        self.vector_dipole = vector_dipole
        self.charge_neutrality = charge_neutrality
        self.regularization = regularization

        input_dim: int = getattr(encoder, "output_dim", 16)
        edge_dim: int = getattr(encoder, "edge_output_dim", input_dim)

        self.uses_ac = "ac" in variant
        self.uses_ad = "ad" in variant
        self.uses_bc = "bc" in variant
        self.uses_os = variant == "ad_os"
        if not (self.uses_ac or self.uses_ad or self.uses_bc or self.uses_os):
            raise ValueError(f"Unsupported PiNet dipole variant {variant!r}.")

        if self.uses_ac:
            self.charge_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
            )
        if self.uses_ad or self.uses_os:
            self.atomic_dipole_gate = nn.Linear(input_dim, 1, bias=False, dtype=config.ftype)
        if self.uses_bc:
            self.bond_scalar_mlp = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
            )
            self.bond_vector_mlp = nn.Linear(input_dim, 1, bias=False, dtype=config.ftype)

    def forward(self, batch: GraphBatch) -> dict[str, torch.Tensor]:
        _recompute_edges(batch)
        batch = self.encoder(batch)

        pos = batch["atoms", "pos"]
        atom_batch = batch["atoms", "batch"]
        num = _num_graphs(atom_batch)
        node_feats = _pool_layer(batch["atoms", "node_features"], self.layer_reduction)

        dipole = torch.zeros(num, 3, dtype=pos.dtype, device=pos.device)
        out: dict[str, torch.Tensor] = {}

        if self.uses_ac:
            charges = self.charge_mlp(node_feats).squeeze(-1)
            total_charge = None
            if self.charge_neutrality:
                try:
                    total_charge = batch["graphs", "total_charge"]
                except KeyError:
                    pass
                charges, pre, post = _charge_neutralize(
                    charges, atom_batch, total_charge=total_charge,
                )
                out["charge_sum_pre_proj"] = pre
                out["charge_sum_post_proj"] = post
            out["atomic_charges"] = charges
            dipole = dipole + _scatter_sum(charges.unsqueeze(-1) * pos, atom_batch, num)

        if self.uses_os:
            oxidation = batch["atoms", "oxidation"].to(dtype=pos.dtype)
            out["oxidation_charges"] = oxidation
            dipole = dipole + _scatter_sum(oxidation.unsqueeze(-1) * pos, atom_batch, num)

        if self.uses_ad or self.uses_os:
            p3 = _pool_layer(batch["atoms", "p3_features"], self.layer_reduction)
            atomic_dipoles = self.atomic_dipole_gate(p3).squeeze(-1)
            out["atomic_dipoles"] = atomic_dipoles
            dipole = dipole + _scatter_sum(atomic_dipoles, atom_batch, num)

        if self.uses_bc:
            edge_index = batch["edges", "edge_index"]
            edge_batch = atom_batch[edge_index[:, 0]]
            i1 = _pool_layer(batch["edges", "i1_features"], self.layer_reduction)
            bond_charge = self.bond_scalar_mlp(i1).squeeze(-1)
            if "i3_features" in batch["edges"].keys():
                i3 = _pool_layer(batch["edges", "i3_features"], self.layer_reduction)
                bond_charge = bond_charge + self.bond_vector_mlp(i3.square().sum(dim=1)).squeeze(-1)
            out["bond_charges"] = bond_charge
            bond_dipoles = bond_charge.unsqueeze(-1) * batch["edges", "bond_diff"]
            out["bond_dipoles"] = bond_dipoles
            dipole = dipole + _scatter_sum(bond_dipoles, edge_batch, num)
            if self.regularization:
                out["bond_charge_l2"] = bond_charge.square().mean()

        out["molecular_dipole"] = dipole
        if self.vector_dipole:
            out["dipole"] = dipole
        else:
            out["dipole"] = torch.sqrt(dipole.square().sum(dim=-1) + 1e-6)
        return out
