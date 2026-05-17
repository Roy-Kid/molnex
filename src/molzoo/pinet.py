"""PiNet encoder.

PiNet is an encoder-only architecture that produces multi-rank representation
tracks (P1 scalar, P3 vector, P5 rank-5). Energy, dipole, charge-response,
and polarizability readouts live in :mod:`molpot`.

Reference:
    Li et al. "PiNN: Equivariant Neural Network Suite for Modeling
    Electrochemical Systems", JCTC 2025.
    https://doi.org/10.1021/acs.jctc.4c01570

    Reference implementation:
    https://github.com/Teoroo-CMC/PiNN/blob/master/pinn/networks/pinet2.py
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from tensordict.nn import TensorDictModuleBase

from molix import config
from molix.data.types import GraphBatch
from molrep.embedding.cutoff import CosineCutoff, HalfCosineCutoff, TanhCutoff
from molrep.embedding.radial import GaussianBasis, PolynomialBasis
from molrep.interaction.pinet import GCBlock, ResUpdate

__all__ = ["PiNet", "PiNetSpec"]


class PiNetSpec(BaseModel):
    """Configuration snapshot for :class:`PiNet`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    atom_types: list[int] = Field(default_factory=lambda: [1, 6, 7, 8], min_length=1)
    r_max: float = Field(default=4.0, gt=0.0)
    cutoff_type: Literal["f1", "f2", "hip"] = "f1"
    basis_type: Literal["polynomial", "gaussian"] = "polynomial"
    n_basis: int = Field(default=4, gt=0)
    gamma: float | list[float] = 3.0
    center: float | list[float] | None = None
    pp_nodes: list[int] = Field(default_factory=lambda: [16, 16], min_length=1)
    pi_nodes: list[int] = Field(default_factory=lambda: [16, 16], min_length=1)
    ii_nodes: list[int] = Field(default_factory=lambda: [16, 16], min_length=1)
    depth: int = Field(default=4, gt=0)
    activation: str = "tanh"
    weighted: bool = False
    rank: Literal[1, 3, 5] = 3


def _compute_d5(d3: torch.Tensor) -> torch.Tensor:
    """Five-component symmetric-traceless rank-5 direction basis."""
    x, y, z = d3[:, 0], d3[:, 1], d3[:, 2]
    x2, y2, z2 = x.square(), y.square(), z.square()
    return torch.stack(
        [
            (2.0 / 3.0) * x2 - (1.0 / 3.0) * y2 - (1.0 / 3.0) * z2,
            (2.0 / 3.0) * y2 - (1.0 / 3.0) * x2 - (1.0 / 3.0) * z2,
            x * y,
            x * z,
            y * z,
        ],
        dim=1,
    )


class PiNet(TensorDictModuleBase):
    """PiNet feature encoder.

    Inputs from the nested ``GraphBatch`` schema:

    * ``("atoms", "Z")``: atomic numbers ``(N,)``.
    * ``("edges", "edge_index")``: source-target pairs ``(E, 2)``.
    * ``("edges", "bond_diff")``: displacements ``pos[target] - pos[source]``.
    * ``("edges", "bond_dist")``: distances ``(E,)``.

    Writes in place:

    * ``("atoms", "node_features")``: scalar P1 states ``(N, depth, D)``.
    * ``("atoms", "p3_features")``: vector P3 states ``(N, depth, 3, D)``.
    * ``("atoms", "p5_features")``: rank-5 P5 states ``(N, depth, 5, D)``.
    * ``("edges", "i1_features")``: scalar pair interactions ``(E, depth, D * n_props)``.
    * ``("edges", "i3_features")`` / ``("edges", "i5_features")`` when enabled.
    """

    in_keys = [
        ("atoms", "Z"),
        ("edges", "edge_index"),
        ("edges", "bond_diff"),
        ("edges", "bond_dist"),
    ]
    out_keys = [("atoms", "node_features")]

    def __init__(
        self,
        *,
        atom_types: list[int] | None = None,
        r_max: float = 4.0,
        cutoff_type: Literal["f1", "f2", "hip"] = "f1",
        basis_type: Literal["polynomial", "gaussian"] = "polynomial",
        n_basis: int = 4,
        gamma: float | list[float] = 3.0,
        center: float | list[float] | None = None,
        pp_nodes: list[int] | None = None,
        pi_nodes: list[int] | None = None,
        ii_nodes: list[int] | None = None,
        depth: int = 4,
        activation: str = "tanh",
        weighted: bool = False,
        rank: Literal[1, 3, 5] = 3,
    ) -> None:
        super().__init__()
        self.config = PiNetSpec(
            atom_types=atom_types or [1, 6, 7, 8],
            r_max=r_max,
            cutoff_type=cutoff_type,
            basis_type=basis_type,
            n_basis=n_basis,
            gamma=gamma,
            center=center,
            pp_nodes=pp_nodes or [16, 16],
            pi_nodes=pi_nodes or [16, 16],
            ii_nodes=ii_nodes or [16, 16],
            depth=depth,
            activation=activation,
            weighted=weighted,
            rank=rank,
        )
        cfg = self.config
        if cfg.pp_nodes[-1] != cfg.ii_nodes[-1]:
            raise ValueError("PiNet requires pp_nodes[-1] == ii_nodes[-1].")

        self.rank = int(cfg.rank)
        self.depth = int(cfg.depth)
        self.feature_dim = int(cfg.ii_nodes[-1])
        self.n_props = int(self.rank // 2) + 1
        self.output_dim = self.feature_dim
        self.edge_output_dim = self.feature_dim * self.n_props

        self.element_embedding = nn.Embedding(len(cfg.atom_types), self.feature_dim)
        self.element_embedding = self.element_embedding.to(dtype=config.ftype)
        self.register_buffer(
            "_atom_types", torch.tensor(cfg.atom_types, dtype=torch.long), persistent=False,
        )
        self._atom_types: torch.Tensor

        _cutoff_cls = {"f1": CosineCutoff, "f2": TanhCutoff, "hip": HalfCosineCutoff}
        self.cutoff = _cutoff_cls[cfg.cutoff_type](r_cut=cfg.r_max)

        if cfg.basis_type == "polynomial":
            self.basis_fn = PolynomialBasis(cfg.n_basis)
        else:
            self.basis_fn = GaussianBasis(
                center=cfg.center, gamma=cfg.gamma, r_cut=cfg.r_max, n_basis=cfg.n_basis,
            )

        self.gc_blocks = torch.nn.ModuleList(
            [
                GCBlock(
                    rank=self.rank,
                    weighted=cfg.weighted,
                    pp_nodes=cfg.pp_nodes,
                    pi_nodes=cfg.pi_nodes,
                    ii_nodes=cfg.ii_nodes,
                    n_basis=cfg.n_basis,
                    activation=cfg.activation,
                )
                for _ in range(self.depth)
            ]
        )

        p1_dims = [self.feature_dim] + [self.feature_dim] * self.depth
        self.res_update1 = torch.nn.ModuleList(
            [ResUpdate(in_dim=p1_dims[i], out_dim=p1_dims[i + 1]) for i in range(self.depth)]
        )
        if self.rank >= 3:
            p3_dims = [1] + [self.feature_dim] * self.depth
            self.res_update3 = torch.nn.ModuleList(
                [ResUpdate(in_dim=p3_dims[i], out_dim=p3_dims[i + 1]) for i in range(self.depth)]
            )
        if self.rank >= 5:
            p5_dims = [1] + [self.feature_dim] * self.depth
            self.res_update5 = torch.nn.ModuleList(
                [ResUpdate(in_dim=p5_dims[i], out_dim=p5_dims[i + 1]) for i in range(self.depth)]
            )

    @classmethod
    def from_spec(cls, spec: PiNetSpec) -> "PiNet":
        return cls(**spec.model_dump())

    def forward(self, td: GraphBatch) -> GraphBatch:
        Z = td["atoms", "Z"]
        edge_index = td["edges", "edge_index"]
        bond_diff = td["edges", "bond_diff"]
        bond_dist = td["edges", "bond_dist"]

        idx = torch.zeros_like(Z)
        for i, z in enumerate(self.config.atom_types):
            idx = torch.where(Z == z, torch.full_like(idx, i), idx)
        p1 = self.element_embedding(idx)
        tensors: dict[str, torch.Tensor] = {"edge_index": edge_index, "p1": p1}

        d3 = bond_diff / bond_dist.clamp(min=1e-8).unsqueeze(-1)
        tensors["d3"] = d3
        if self.rank >= 3:
            tensors["p3"] = torch.zeros(
                Z.shape[0], 3, 1, dtype=bond_diff.dtype, device=bond_diff.device,
            )
        if self.rank >= 5:
            tensors["p5"] = torch.zeros(
                Z.shape[0], 5, 1, dtype=bond_diff.dtype, device=bond_diff.device,
            )
            tensors["d5"] = _compute_d5(d3)

        fc = self.cutoff(bond_dist)
        basis = self.basis_fn(bond_dist, fc=fc)

        p1_states: list[torch.Tensor] = []
        p3_states: list[torch.Tensor] = []
        p5_states: list[torch.Tensor] = []
        i1_states: list[torch.Tensor] = []
        i3_states: list[torch.Tensor] = []
        i5_states: list[torch.Tensor] = []

        for i, block in enumerate(self.gc_blocks):
            new = block(tensors, basis)
            tensors["p1"] = self.res_update1[i](tensors["p1"], new["p1"])
            p1_states.append(tensors["p1"])
            i1_states.append(new["i1"])

            if self.rank >= 3:
                tensors["p3"] = self.res_update3[i](tensors["p3"], new["p3"])
                p3_states.append(tensors["p3"])
                i3_states.append(new["i3"])

            if self.rank >= 5:
                tensors["p5"] = self.res_update5[i](tensors["p5"], new["p5"])
                p5_states.append(tensors["p5"])
                i5_states.append(new["i5"])

        td["atoms", "node_features"] = torch.stack(p1_states, dim=1)
        td["edges", "i1_features"] = torch.stack(i1_states, dim=1)
        if self.rank >= 3:
            td["atoms", "p3_features"] = torch.stack(p3_states, dim=1)
            td["edges", "i3_features"] = torch.stack(i3_states, dim=1)
        if self.rank >= 5:
            td["atoms", "p5_features"] = torch.stack(p5_states, dim=1)
            td["edges", "i5_features"] = torch.stack(i5_states, dim=1)
        return td
