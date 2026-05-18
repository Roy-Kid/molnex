"""PiNet encoder + potential head.

PiNet is a multi-rank representation architecture (P1 scalar, P3 vector, P5
rank-5). The encoder produces ``(N, layers, features)`` node features; the
PiNet-specific head pools across layers, predicts per-atom energies, and
derives forces via autograd. The head lives alongside the encoder (not under
``molpot``) because the pool-then-readout pattern only makes sense for PiNet's
multi-layer output shape — there is no MACE/Allegro reuse to extract.

Dipole and polarizability readouts live in :mod:`molpot` (under
``pinet_dipole`` / ``pinet_polarizability``) because they pair the encoder
with task-specific tensor heads that don't compose with the energy head.

Reference:
    Li et al. "PiNN: Equivariant Neural Network Suite for Modeling
    Electrochemical Systems", JCTC 2025.
    https://doi.org/10.1021/acs.jctc.4c01570

    Reference implementation:
    https://github.com/Teoroo-CMC/PiNN/blob/master/pinn/networks/pinet2.py
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from molix import config
from molpot.derivation import EnergyAggregation, ForceDerivation
from molpot.heads import ChargeResponseHead, DipoleHead
from molrep.embedding.cutoff import CosineCutoff, HalfCosineCutoff, TanhCutoff
from molrep.embedding.radial import GaussianBasis, PolynomialBasis
from molrep.interaction.pinet import GCBlock, ResUpdate

__all__ = [
    "PiNet",
    "PiNetSpec",
    "PiNetPotential",
    "PiNetDipole",
    "PiNetPolarizability",
]


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

    Inputs from the nested ``TensorDict`` schema:

    * ``("atoms", "Z")``: atomic numbers ``(N,)``.
    * ``("atoms", "pos")``: positions ``(N, 3)``.
    * ``("edges", "edge_index")``: source-target pairs ``(E, 2)``.

    Edge geometry (``bond_diff`` = ``pos[target] - pos[source]`` and
    ``bond_dist`` = ``‖bond_diff‖``) is derived from ``pos`` + ``edge_index``
    inside ``forward``. Caching them in the pipeline buys nothing (one
    gather + diff + norm per step) and breaks autograd-through-pos for
    force training.

    Writes in place:

    * ``("atoms", "node_features")``: scalar P1 states ``(N, depth, D)``.
    * ``("atoms", "p3_features")``: vector P3 states ``(N, depth, 3, D)``.
    * ``("atoms", "p5_features")``: rank-5 P5 states ``(N, depth, 5, D)``.
    * ``("edges", "i1_features")``: scalar pair interactions ``(E, depth, D * n_props)``.
    * ``("edges", "i3_features")`` / ``("edges", "i5_features")`` when enabled.
    """

    in_keys = [
        ("atoms", "Z"),
        ("atoms", "pos"),
        ("edges", "edge_index"),
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
            "_atom_types",
            torch.tensor(cfg.atom_types, dtype=torch.long),
            persistent=False,
        )
        self._atom_types: torch.Tensor

        # Z -> embedding-row lookup table. Sized to cover every stable element
        # so a single gather replaces the per-forward Python loop. Unknown Z
        # falls back to row 0, matching the prior masked_fill_ default.
        z_to_idx = torch.zeros(128, dtype=torch.long)
        for i, z in enumerate(cfg.atom_types):
            z_to_idx[int(z)] = i
        self.register_buffer("_z_to_idx", z_to_idx, persistent=False)
        self._z_to_idx: torch.Tensor

        _cutoff_cls = {"f1": CosineCutoff, "f2": TanhCutoff, "hip": HalfCosineCutoff}
        self.cutoff = _cutoff_cls[cfg.cutoff_type](r_cut=cfg.r_max)

        if cfg.basis_type == "polynomial":
            self.basis_fn = PolynomialBasis(cfg.n_basis)
        else:
            self.basis_fn = GaussianBasis(
                center=cfg.center,
                gamma=cfg.gamma,
                r_cut=cfg.r_max,
                n_basis=cfg.n_basis,
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

    def forward(self, td: TensorDict) -> TensorDict:
        Z = td["atoms", "Z"]
        pos = td["atoms", "pos"]
        edge_index = td["edges", "edge_index"]

        # Edge geometry derived from pos + edge_index per forward.
        # One gather + diff + norm — cheaper than caching, and gradients flow
        # back to pos naturally so force training needs no special path.
        bond_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
        bond_dist = bond_diff.norm(dim=-1).clamp(min=1e-8)

        idx = self._z_to_idx[Z]
        p1 = self.element_embedding(idx)
        tensors: dict[str, torch.Tensor] = {"edge_index": edge_index, "p1": p1}

        d3 = bond_diff / bond_dist.unsqueeze(-1)
        tensors["d3"] = d3
        if self.rank >= 3:
            tensors["p3"] = torch.zeros(
                Z.shape[0],
                3,
                1,
                dtype=bond_diff.dtype,
                device=bond_diff.device,
            )
        if self.rank >= 5:
            tensors["p5"] = torch.zeros(
                Z.shape[0],
                5,
                1,
                dtype=bond_diff.dtype,
                device=bond_diff.device,
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


# ---------------------------------------------------------------------------
# PiNet energy + force potential
# ---------------------------------------------------------------------------


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

    Pools the encoder's ``(N, layers, features)`` output across the layer
    axis, predicts per-atom energies via a 2-layer MLP, aggregates them to
    graph energies, and derives forces via ``torch.autograd.grad`` when
    ``compute_forces`` (or ``compute_forces_default``) is set.

    Per-element baseline subtraction (atomic dress) is handled at cache
    time by :class:`molix.data.AtomicDress` — labels are already dressed
    when they reach this model, so no runtime dress add is needed.

    Args:
        encoder: :class:`PiNet` (or any module that writes ``("atoms",
            "node_features")`` into a ``TensorDict``).
        hidden_dim: Hidden dimension of the per-atom energy MLP.
        layer_reduction: How to pool across GC-block layers
            (``"mean"`` / ``"sum"`` / ``"last"``).
        compute_forces: Default value for forward's ``compute_forces``
            kwarg. Set to ``True`` for force training so the Trainer's plain
            ``model(batch)`` call returns ``{"energy", "forces"}``.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        hidden_dim: int = 64,
        layer_reduction: Literal["mean", "sum", "last"] = "mean",
        compute_forces: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.layer_reduction = layer_reduction
        self.compute_forces_default = compute_forces

        input_dim: int = getattr(encoder, "output_dim", 16)
        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=config.ftype),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, dtype=config.ftype),
        )
        self.energy_aggregation = EnergyAggregation(pooling="sum")
        self.force_derivation = ForceDerivation()

    def forward(
        self, batch: TensorDict, *, compute_forces: bool | None = None
    ) -> dict[str, torch.Tensor]:
        if compute_forces is None:
            compute_forces = self.compute_forces_default
        grad_ctx = torch.enable_grad() if compute_forces else nullcontext()
        with grad_ctx:
            if compute_forces and not batch["atoms", "pos"].requires_grad:
                batch["atoms", "pos"] = batch["atoms", "pos"].clone().requires_grad_(True)
            out = self._energy_forward(batch)
            if compute_forces:
                out["forces"] = self.force_derivation(out["energy"], batch["atoms", "pos"])
        return out

    def _energy_forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        """Compilable energy computation: encoder -> pool -> head -> aggregate.

        Extracted so ``torch.compile`` can wrap it without hitting the
        double-backward limitation that ``ForceDerivation``'s internal
        ``torch.autograd.grad`` triggers.
        """
        batch = self.encoder(batch)

        node_feats = _pool_layer(batch["atoms", "node_features"], self.layer_reduction)
        atom_energy = self.node_mlp(node_feats).squeeze(-1)

        atom_batch = batch["atoms", "batch"]
        num_graphs = batch["graphs"].batch_size[0]
        energy = self.energy_aggregation(atom_energy, atom_batch, num_graphs=num_graphs)

        return {
            "atomic_energy": atom_energy,
            "energy": energy,
        }

    def compile_encoder(self, *, backend: str = "inductor", **kwargs) -> None:
        """Compile only the encoder, leaving the head and force derivation eager.

        ``torch.compile`` on the full :class:`PiNetPotential` is not supported
        for force training because :class:`ForceDerivation` calls
        ``torch.autograd.grad`` internally, which triggers a double-backward
        through the compiled graph that ``aot_autograd`` cannot handle.

        For energy-only training, ``torch.compile(potential)`` works fine.
        For energy+force training, call ``potential.compile_encoder()``
        before the Trainer — this compiles the encoder (95%+ of compute)
        while leaving the ``autograd.grad`` path eager.
        """
        self.encoder = torch.compile(self.encoder, backend=backend, **kwargs)


# ---------------------------------------------------------------------------
# PiNet + DipoleHead composed model
# ---------------------------------------------------------------------------


class PiNetDipole(nn.Module):
    """PiNet encoder paired with :class:`molpot.heads.DipoleHead`.

    The encoder writes scalar/vector tracks ``(N, layers, ...)`` into the
    batch; this wrapper pools across the layer axis and forwards the
    pre-pooled tensors to a generic dipole head. See
    :class:`molpot.heads.DipoleHead` for variant semantics.
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
        self.encoder = encoder
        self.layer_reduction = layer_reduction
        input_dim: int = getattr(encoder, "output_dim", 16)
        edge_dim: int = getattr(encoder, "edge_output_dim", input_dim)
        self.head = DipoleHead(
            node_scalar_dim=input_dim,
            node_vector_dim=input_dim,
            edge_scalar_dim=edge_dim,
            edge_vector_dim=input_dim,
            hidden_dim=hidden_dim,
            variant=variant,
            vector_dipole=vector_dipole,
            charge_neutrality=charge_neutrality,
            regularization=regularization,
        )

    def forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        batch = self.encoder(batch)

        atom_batch = batch["atoms", "batch"]
        num_graphs = batch["graphs"].batch_size[0]
        node_scalars = _pool_layer(
            batch["atoms", "node_features"],
            self.layer_reduction,
        )
        node_vectors = None
        if "p3_features" in batch["atoms"].keys():
            node_vectors = _pool_layer(
                batch["atoms", "p3_features"],
                self.layer_reduction,
            )
        edge_scalars = None
        edge_index = None
        bond_diff = None
        if self.head.uses_bc and "i1_features" in batch["edges"].keys():
            edge_scalars = _pool_layer(
                batch["edges", "i1_features"],
                self.layer_reduction,
            )
            edge_index = batch["edges", "edge_index"]
            pos = batch["atoms", "pos"]
            bond_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
        edge_vectors = None
        if self.head.uses_bc and "i3_features" in batch["edges"].keys():
            edge_vectors = _pool_layer(
                batch["edges", "i3_features"],
                self.layer_reduction,
            )
        oxidation = None
        if self.head.uses_os and "oxidation" in batch["atoms"].keys():
            oxidation = batch["atoms", "oxidation"]
        total_charge = None
        if self.head.uses_ac and self.head.charge_neutrality:
            try:
                total_charge = batch["graphs", "total_charge"]
            except KeyError:
                pass
        return self.head(
            pos=batch["atoms", "pos"],
            atom_batch=atom_batch,
            num_graphs=num_graphs,
            node_scalars=node_scalars,
            node_vectors=node_vectors,
            edge_scalars=edge_scalars,
            edge_vectors=edge_vectors,
            edge_index=edge_index,
            bond_diff=bond_diff,
            oxidation=oxidation,
            total_charge=total_charge,
        )


# ---------------------------------------------------------------------------
# PiNet + ChargeResponseHead composed model
# ---------------------------------------------------------------------------


class PiNetPolarizability(nn.Module):
    """PiNet encoder paired with :class:`molpot.heads.ChargeResponseHead`.

    See :class:`molpot.heads.ChargeResponseHead` for variant semantics
    (``localchi`` / ``local`` / ``etainv`` / ``eem`` / ``acks2``).
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        atom_types: list[int] | None = None,
        variant: str = "localchi",
        iso: bool = False,
        hidden_dim: int = 64,
        layer_reduction: Literal["mean", "sum", "last"] = "mean",
        epsilon: float = 0.01,
        sigma: dict[int, float] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.layer_reduction = layer_reduction
        input_dim: int = getattr(encoder, "output_dim", 16)
        edge_dim: int = getattr(encoder, "edge_output_dim", input_dim)
        self.head = ChargeResponseHead(
            node_scalar_dim=input_dim,
            edge_scalar_dim=edge_dim,
            edge_vector_dim=input_dim,
            atom_types=atom_types,
            variant=variant,
            iso=iso,
            hidden_dim=hidden_dim,
            epsilon=epsilon,
            sigma=sigma,
        )

    def forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        batch = self.encoder(batch)

        atom_batch = batch["atoms", "batch"]
        num_graphs = batch["graphs"].batch_size[0]
        node_scalars = _pool_layer(
            batch["atoms", "node_features"],
            self.layer_reduction,
        )
        edge_scalars = _pool_layer(
            batch["edges", "i1_features"],
            self.layer_reduction,
        )
        edge_vectors = None
        if "i3_features" in batch["edges"].keys():
            edge_vectors = _pool_layer(
                batch["edges", "i3_features"],
                self.layer_reduction,
            )
        pos = batch["atoms", "pos"]
        edge_index = batch["edges", "edge_index"]
        bond_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
        return self.head(
            pos=pos,
            Z=batch["atoms", "Z"],
            atom_batch=atom_batch,
            num_graphs=num_graphs,
            edge_index=edge_index,
            bond_diff=bond_diff,
            node_scalars=node_scalars,
            edge_scalars=edge_scalars,
            edge_vectors=edge_vectors,
        )
