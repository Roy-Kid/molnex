"""Edge-centric energy readout for Allegro-style encoders.

Separates three responsibilities that used to be entangled in training scripts:

1. **Layer reduction** — stacked per-layer edge features ``(E, L, F)`` →
   ``(E, F)``.
2. **Per-edge energy** — linear readout MLP with *no* trailing activation,
   matching the paper's ``edge_eng`` module (reference:
   ``mir-group/allegro::_fc.py::ScalarMLPFunction`` with
   ``nonlinearity=None`` and ``edge_eng_mlp_latent_dimensions=[128]``).
3. **Aggregation** — scatter to source atom with ``1/√⟨|N(i)|⟩`` normalization
   (paper SI "Normalization"), then sum atoms to graph.

This keeps the encoder truly encoder-only and lets callers swap the readout
or aggregation without touching Allegro internals.

Reference:
    Musaelian et al. "Learning Local Equivariant Representations for
    Large-Scale Atomistic Dynamics" Nature Communications 14, 579 (2023)
    https://arxiv.org/abs/2204.05249

    mir-group/allegro/nn/_edgewise.py::EdgewiseEnergySum
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from molix import config
from molix.data.types import GraphBatch


class EdgeEnergyHead(nn.Module):
    """Map per-edge scalar features to total per-graph energy.

    Pipeline::

        edge_features (E, L, F)  —— reduce L ——→  (E, F)
        (E, F)                   —— Linear(F, H) → Linear(H, 1) ——→  e_ij (E,)
        e_ij                     —— scatter_add by source, / √⟨N⟩ ——→  E_i (N,)
        E_i                      —— scatter_add by batch ——→  E (B,)

    The readout MLP is intentionally fully linear (no activation on the hidden
    layer either — only two linear layers in series). This matches
    ``mir-group/allegro::minimal.yaml`` where
    ``edge_eng_mlp_nonlinearity=null`` and ``edge_eng_mlp_latent_dimensions=[128]``.

    Args:
        input_dim: Per-edge feature dim ``F`` (= ``num_scalar_features`` of the
            encoder).
        hidden_dim: Readout hidden dim ``H``. Paper QM9: ``128``.
        avg_num_neighbors: Dataset-wide ⟨|N(i)|⟩ for neighbour normalization.
            Must be positive. Falls back to per-source ``1/√|N(i)|`` when set
            to ``None``.
        layer_reduction: Strategy for reducing the stacked per-layer axis ``L``.

            * ``"last"`` — use the final layer's scalar (paper default for the
              ``edge_eng`` readout).
            * ``"mean"`` / ``"sum"`` — average / sum across layers.

        out_key: TensorDict key under which to write the per-graph energy.

    Returns:
        ``{out_key: (B,)}`` — per-graph total energy tensor.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        avg_num_neighbors: float | None,
        layer_reduction: Literal["last", "mean", "sum"] = "last",
        out_key: str = "energy",
    ):
        super().__init__()
        if layer_reduction not in ("last", "mean", "sum"):
            raise ValueError(
                f"layer_reduction must be 'last', 'mean', or 'sum'; got '{layer_reduction}'"
            )
        if avg_num_neighbors is not None and avg_num_neighbors <= 0.0:
            raise ValueError(
                f"avg_num_neighbors must be positive or None; got {avg_num_neighbors}"
            )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=config.ftype),
            nn.Linear(hidden_dim, 1, dtype=config.ftype),
        )
        self.layer_reduction = layer_reduction
        self.avg_num_neighbors = (
            float(avg_num_neighbors) if avg_num_neighbors is not None else None
        )
        self.out_key = out_key

    def _reduce_layers(self, edge_features: torch.Tensor) -> torch.Tensor:
        if edge_features.ndim == 2:
            return edge_features
        if edge_features.ndim != 3:
            raise ValueError(
                f"edge_features must be 2D or 3D, got {edge_features.ndim}D"
            )
        if self.layer_reduction == "last":
            return edge_features[:, -1]
        if self.layer_reduction == "mean":
            return edge_features.mean(dim=1)
        return edge_features.sum(dim=1)

    def forward(self, batch: GraphBatch) -> dict[str, torch.Tensor]:
        """Aggregate edge features to per-graph energies.

        Args:
            batch: ``GraphBatch`` with ``edges.edge_features``,
                ``edges.edge_index``, ``atoms.Z``, ``atoms.batch``, and a
                ``graphs`` subdict that carries the per-graph batch size.

        Returns:
            ``{self.out_key: Tensor}`` with per-graph energy of shape ``(B,)``.
        """
        edge_features = batch["edges", "edge_features"]
        edge_features = self._reduce_layers(edge_features)
        edge_e = self.mlp(edge_features).squeeze(-1)                  # (E,)

        if self.avg_num_neighbors is not None:
            edge_e = edge_e / math.sqrt(self.avg_num_neighbors)
        else:
            edge_e = self._per_source_sqrt_norm(edge_e, batch)

        edge_index = batch["edges", "edge_index"]
        n_nodes = int(batch["atoms", "Z"].shape[0])
        node_e = torch.zeros(
            n_nodes, dtype=edge_e.dtype, device=edge_e.device
        )
        node_e.scatter_add_(0, edge_index[:, 0], edge_e)              # (N,)

        atom_batch = batch["atoms", "batch"]
        n_graphs = batch["graphs"].batch_size[0]
        energy = torch.zeros(
            n_graphs, dtype=node_e.dtype, device=node_e.device
        )
        energy.scatter_add_(0, atom_batch, node_e)                    # (B,)

        return {self.out_key: energy}

    @staticmethod
    def _per_source_sqrt_norm(
        edge_e: torch.Tensor, batch: GraphBatch
    ) -> torch.Tensor:
        """Per-source ``1/√|N(i)|`` fallback when no dataset constant is given."""
        edge_index = batch["edges", "edge_index"]
        n_nodes = int(batch["atoms", "Z"].shape[0])
        src = edge_index[:, 0]
        src_count = torch.zeros(
            n_nodes, dtype=edge_e.dtype, device=edge_e.device
        )
        src_count.scatter_add_(
            0,
            src,
            torch.ones(src.shape[0], dtype=edge_e.dtype, device=edge_e.device),
        )
        return edge_e / src_count.clamp(min=1.0).sqrt()[src]
