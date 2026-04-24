"""Allegro: local equivariant edge encoder.

Pair-level equivariant encoder using iterated tensor products with neighbourhood
aggregation. Returns per-layer per-edge scalar features only — readout, energy
aggregation, and any rescaling live in :mod:`molpot.heads`.

This module is a **redesign** (2026-04): it matches the paper's residual update
exactly (``u(r_ij)`` gates the MLP update inside every layer), drops the
post-layer-loop cutoff hack, drops the extra post-TP tensor-track rescaling,
and removes the spurious trailing SiLU on the scalar MLP. See
``src/molzoo/specs/allegro.md`` for the full math.

Reference:
    Musaelian et al. "Learning Local Equivariant Representations for
    Large-Scale Atomistic Dynamics" Nature Communications 14, 579 (2023)
    https://arxiv.org/abs/2204.05249
"""

from __future__ import annotations

import math

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn
from cuequivariance import O3, Irreps
from pydantic import BaseModel, ConfigDict, Field
from tensordict.nn import TensorDictModuleBase

from molix import config
from molix.data.types import GraphBatch
from molrep.embedding.angular import SphericalHarmonics
from molrep.embedding.cutoff import PolynomialCutoff
from molrep.embedding.radial import BesselRBF
from molrep.interaction.tensor_product import irreps_from_l_max, sh_irreps_from_l_max


# ===========================================================================
# Helpers — pure functions on ir_mul-layout tensors
# ===========================================================================


def _env_weight_harmonics(
    edge_angular: torch.Tensor,
    env_weights: torch.Tensor,
    l_max: int,
    num_tensor_features: int,
    irreps_dim: int,
) -> torch.Tensor:
    """Weight spherical harmonics by per-channel environment weights.

    For each angular momentum ``l``, outer-products the ``(2l+1)`` spherical
    harmonic components with the ``num_tensor_features`` channel weights in
    ``ir_mul`` layout.

    Args:
        edge_angular: Spherical harmonics ``(n_edges, sh_dim)``.
        env_weights: Per-channel weights ``(n_edges, num_tensor_features)``.
        l_max: Maximum angular momentum.
        num_tensor_features: Number of tensor feature channels.
        irreps_dim: Total dimension of the ir_mul tensor representation.

    Returns:
        Weighted tensor features ``(n_edges, irreps_dim)`` in ir_mul layout.
    """
    out = torch.zeros(
        edge_angular.shape[0],
        irreps_dim,
        dtype=edge_angular.dtype,
        device=edge_angular.device,
    )
    off_sh = 0
    off_tp = 0
    for l in range(l_max + 1):
        deg = 2 * l + 1
        ylm = edge_angular[:, off_sh : off_sh + deg]
        block = (ylm.unsqueeze(-1) * env_weights.unsqueeze(-2)).reshape(
            ylm.shape[0], -1
        )
        out[:, off_tp : off_tp + deg * num_tensor_features] = block
        off_sh += deg
        off_tp += deg * num_tensor_features
    return out


def _scale_by_channel(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    l_max: int,
    num_tensor_features: int,
) -> torch.Tensor:
    """Scale an ir_mul tensor feature by per-channel scalar weights.

    Args:
        tensor: Equivariant tensor in ir_mul layout ``(n_edges, irreps_dim)``.
        scale: Per-channel scale factors ``(n_edges, num_tensor_features)``.
        l_max: Maximum angular momentum.
        num_tensor_features: Number of tensor feature channels.

    Returns:
        Scaled tensor ``(n_edges, irreps_dim)`` in ir_mul layout.
    """
    out = torch.empty_like(tensor)
    offset = 0
    for l in range(l_max + 1):
        deg = 2 * l + 1
        bsz = deg * num_tensor_features
        block = tensor[:, offset : offset + bsz].reshape(-1, deg, num_tensor_features)
        out[:, offset : offset + bsz] = (block * scale.unsqueeze(-2)).reshape(-1, bsz)
        offset += bsz
    return out


def _make_scalar_mlp(
    in_dim: int,
    hiddens: list[int],
    out_dim: int,
    activation: type[nn.Module] | None = nn.SiLU,
) -> nn.Sequential:
    """Build ``[Linear→act] * len(hiddens) → Linear`` (paper ScalarMLP layout).

    The final linear has **no** activation (matches
    ``mir-group/allegro::ScalarMLPFunction``). Pass ``activation=None`` to get
    a pure linear stack.
    """
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hiddens:
        layers.append(nn.Linear(prev, h, dtype=config.ftype))
        if activation is not None:
            layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, out_dim, dtype=config.ftype))
    return nn.Sequential(*layers)


# ===========================================================================
# PairEmbedding — two-body radial-chemical embedding
# ===========================================================================


class PairEmbedding(nn.Module):
    """Two-body radial-chemical embedding on edges.

    Architecture::

        bond_dist → BesselRBF * PolynomialCutoff → edge_radial
        edge_dir  → SphericalHarmonics → edge_angular
        Z_i, Z_j  → [Embed(Z_i) ⊕ Embed(Z_j)] → type_embed   (concat)
        (edge_radial ⊕ type_embed) → ScalarMLP → scalar_features
        scalar_features → Linear → tensor_env_weights
        tensor_env_weights ⊗ edge_angular → tensor_features  (ir_mul layout)

    The scalar MLP follows paper convention: ``[Linear→SiLU] × k → Linear``
    with no activation after the final linear. Reference:
    ``mir-group/allegro/nn/_fc.py::ScalarMLPFunction``.

    Args:
        num_elements: Size of the atom-type embedding table (``Z`` is used
            directly as index, so this must be ``> max Z`` in the dataset).
        num_scalar_features: Output scalar feature dim ``F_s``.
        num_tensor_features: Tensor feature channel count ``u``.
        r_max: Cutoff radius (Å).
        num_bessel: Bessel basis size.
        l_max: Maximum angular momentum for spherical harmonics.
        type_emb_dim: Dim of each per-atom type embedding (concatenated source
            and destination embeddings have total dim ``2 * type_emb_dim``).
        scalar_mlp_hiddens: Hidden dims of the 2-body scalar MLP. Paper QM9:
            ``[128, 256, 512]`` (then final ``Linear → num_scalar_features``).
        poly_p: Polynomial cutoff exponent.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        num_scalar_features: int,
        num_tensor_features: int,
        r_max: float,
        num_bessel: int = 8,
        l_max: int = 2,
        type_emb_dim: int = 16,
        scalar_mlp_hiddens: list[int] | None = None,
        poly_p: int = 6,
    ):
        super().__init__()

        self.num_scalar_features = num_scalar_features
        self.num_tensor_features = num_tensor_features
        self.l_max = l_max

        self.radial_basis = BesselRBF(r_cut=r_max, num_radial=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_cut=r_max, exponent=poly_p)
        self.spherical_harmonics = SphericalHarmonics(l_max=l_max)

        self.type_embedding = nn.Embedding(
            num_elements, type_emb_dim, dtype=config.ftype
        )

        scalar_in_dim = num_bessel + 2 * type_emb_dim
        if scalar_mlp_hiddens is None:
            scalar_mlp_hiddens = [num_scalar_features, num_scalar_features]
        self.scalar_mlp = _make_scalar_mlp(
            in_dim=scalar_in_dim,
            hiddens=list(scalar_mlp_hiddens),
            out_dim=num_scalar_features,
            activation=nn.SiLU,
        )

        with cue.assume(O3):
            self.irreps_dim = Irreps(
                irreps_from_l_max(l_max, num_tensor_features)
            ).dim
        self.tensor_env = nn.Linear(
            num_scalar_features, num_tensor_features, dtype=config.ftype
        )

    def forward(
        self,
        Z: torch.Tensor,
        bond_dist: torch.Tensor,
        bond_diff: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute initial pair embeddings.

        Args:
            Z: Atomic numbers ``(n_nodes,)``.
            bond_dist: Bond distances ``(n_edges,)``.
            bond_diff: Bond vectors ``(n_edges, 3)`` (pos[target] - pos[source]).
            edge_index: Edge indices ``(n_edges, 2)``, ``[:, 0]=source``.

        Returns:
            ``(scalar_features, tensor_features, edge_angular, edge_cutoff)``
            with shapes ``(E, F_s)``, ``(E, irreps_dim)``, ``(E, sh_dim)``,
            ``(E,)``.
        """
        src, dst = edge_index[:, 0], edge_index[:, 1]

        edge_radial = self.radial_basis(bond_dist)
        edge_cutoff = self.cutoff_fn(bond_dist)
        edge_radial = edge_radial * edge_cutoff.unsqueeze(-1)

        edge_dir = bond_diff / (bond_dist.unsqueeze(-1) + 1e-8)
        edge_angular = self.spherical_harmonics(edge_dir)

        type_src = self.type_embedding(Z[src])
        type_dst = self.type_embedding(Z[dst])
        type_embed = torch.cat([type_src, type_dst], dim=-1)

        scalar_in = torch.cat([edge_radial, type_embed], dim=-1)
        scalar_features = self.scalar_mlp(scalar_in)

        env_weights = self.tensor_env(scalar_features)
        tensor_features = _env_weight_harmonics(
            edge_angular,
            env_weights,
            self.l_max,
            self.num_tensor_features,
            self.irreps_dim,
        )

        return scalar_features, tensor_features, edge_angular, edge_cutoff


# ===========================================================================
# AllegroLayer — single TP-based layer with cutoff-gated residual
# ===========================================================================


class AllegroLayer(nn.Module):
    """Pair-level tensor product layer with cutoff-gated residual update.

    Flow, for each edge ``(i, j)``:

    1. ``env_w_ij = Linear(scalar_ij)`` → per-channel env weight ``(E, u)``.
    2. ``V_scaled_ij = env_w_ij · V_ij`` (channel-wise scale).
    3. Aggregate single-channel neighbourhood ``Y_i = Σ_{k∈N(i)} Y_ik``,
       normalized by ``1/√avg_num_neighbors`` (paper SI) with per-sample
       ``1/√|N(i)|`` fallback if the dataset constant is not supplied.
    4. ``new_tensor_ij = Linear(TP(V_scaled_ij, Y_i))``   (paper eq. 21).
    5. ``invariants_ij = new_tensor_ij[:, :u]``           (L=0 scalars).
    6. ``mlp_out_ij = u(r_ij) · LatentMLP([scalar_ij, invariants_ij])``
       — **the cutoff gate is applied per-layer on the MLP output, matching
       ``mir-group/allegro::_allegro.py:494``**.
    7. ``scalar_ij ← a · scalar_ij + b · mlp_out_ij`` with
       ``a = 1/√(1+α²), b = α/√(1+α²)``  (paper SI α-residual, α=0.5).
    8. ``V_ij ← new_tensor_ij``  (no post-TP rescale — matches paper).

    The single-channel right operand in step 4 lets us use the fast
    ``ChannelWiseTensorProduct`` kernel; see ``specs/allegro.md §7`` for why
    the bilinearity of the CG tensor product makes this physically equivalent
    to the paper's per-channel-weighted formulation.

    Args:
        num_scalar_features: Scalar feature dim ``F_s``.
        num_tensor_features: Tensor feature channel count ``u``.
        l_max: Maximum angular momentum.
        latent_mlp_hiddens: Hidden dims of the latent MLP.
        latent_activation: Activation class for the latent MLP; pass ``None``
            for a pure linear stack (3BPA setup).
        avg_num_neighbors: Dataset-wide ⟨|N(i)|⟩. ``None`` → per-sample fallback.
        residual_alpha: Residual mixing weight α (paper default 0.5).
    """

    def __init__(
        self,
        *,
        num_scalar_features: int,
        num_tensor_features: int,
        l_max: int = 2,
        latent_mlp_hiddens: list[int] | None = None,
        latent_activation: type[nn.Module] | None = nn.SiLU,
        avg_num_neighbors: float | None = None,
        residual_alpha: float = 0.5,
    ):
        super().__init__()

        self.num_scalar_features = num_scalar_features
        self.num_tensor_features = num_tensor_features
        self.l_max = l_max
        self.avg_num_neighbors = (
            float(avg_num_neighbors) if avg_num_neighbors is not None else None
        )

        irreps_str = irreps_from_l_max(l_max, num_tensor_features)
        sh_irreps_str = sh_irreps_from_l_max(l_max)
        cue_irreps_in = cue.Irreps("O3", irreps_str)
        cue_irreps_sh = cue.Irreps("O3", sh_irreps_str)

        self.tp = cuet.ChannelWiseTensorProduct(
            cue_irreps_in,
            cue_irreps_sh,
            layout=cue.ir_mul,
            shared_weights=True,
            internal_weights=True,
            dtype=config.ftype,
        )
        self.tp_linear = cuet.Linear(
            irreps_in=self.tp.irreps_out,
            irreps_out=cue_irreps_in,
            layout=cue.ir_mul,
            dtype=config.ftype,
        )

        # Pre-TP env embed: current scalars → per-channel V scale.
        self.env_embed = nn.Linear(
            num_scalar_features, num_tensor_features, dtype=config.ftype
        )

        # Latent MLP: [x, invariants] → scalar update.  Final layer linear.
        hiddens = list(latent_mlp_hiddens) if latent_mlp_hiddens else []
        self.latent_mlp = _make_scalar_mlp(
            in_dim=num_scalar_features + num_tensor_features,
            hiddens=hiddens,
            out_dim=num_scalar_features,
            activation=latent_activation,
        )

        # α-residual mixing (paper SI).
        alpha = float(residual_alpha)
        denom = math.sqrt(1.0 + alpha * alpha)
        self.residual_a = 1.0 / denom
        self.residual_b = alpha / denom

        with cue.assume(O3):
            self.irreps_dim = Irreps(irreps_str).dim
            self._sh_dim = Irreps(sh_irreps_str).dim

    def forward(
        self,
        scalar_features: torch.Tensor,
        tensor_features: torch.Tensor,
        edge_angular: torch.Tensor,
        edge_cutoff: torch.Tensor,
        edge_index: torch.Tensor,
        n_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one Allegro layer.

        Args:
            scalar_features: ``(n_edges, F_s)`` current scalar track.
            tensor_features: ``(n_edges, irreps_dim)`` current tensor track.
            edge_angular: ``(n_edges, sh_dim)`` spherical harmonics from the
                embedding (reused across layers).
            edge_cutoff: ``(n_edges,)`` polynomial-cutoff factor ``u(r_ij)``.
            edge_index: ``(n_edges, 2)`` with ``[:,0]=source``.
            n_nodes: Number of nodes in the batch.

        Returns:
            ``(scalar_out, tensor_out)`` with the same shapes as the inputs.
            ``scalar_out`` decays smoothly to ``a · scalar_in`` as
            ``r → r_cut`` because the MLP update is gated by ``u(r_ij)``.
        """
        src = edge_index[:, 0]

        # (1-2) Pre-scale V by env weights.
        env_w = self.env_embed(scalar_features)                       # (E, u)
        scaled_V = _scale_by_channel(
            tensor_features, env_w, self.l_max, self.num_tensor_features
        )                                                             # (E, irreps)

        # (3) Neighbour aggregation of spherical harmonics (single channel).
        node_Y = torch.zeros(
            n_nodes,
            self._sh_dim,
            dtype=tensor_features.dtype,
            device=tensor_features.device,
        )
        node_Y.scatter_add_(
            0, src.unsqueeze(-1).expand_as(edge_angular), edge_angular
        )
        if self.avg_num_neighbors is not None:
            node_Y = node_Y / math.sqrt(self.avg_num_neighbors)
        else:
            src_count = torch.zeros(
                n_nodes,
                dtype=tensor_features.dtype,
                device=tensor_features.device,
            )
            src_count.scatter_add_(
                0,
                src,
                torch.ones(
                    src.shape[0],
                    dtype=tensor_features.dtype,
                    device=tensor_features.device,
                ),
            )
            node_Y = node_Y / src_count.clamp(min=1.0).sqrt().unsqueeze(-1)
        edge_node_Y = node_Y[src]                                     # (E, sh_dim)

        # (4) Tensor product → project back to input irreps.
        tp_out = self.tp(scaled_V, edge_node_Y)
        new_tensor = self.tp_linear(tp_out)                           # (E, irreps)

        # (5) Extract L=0 invariants.
        invariants = new_tensor[:, : self.num_tensor_features]        # (E, u)

        # (6) Cutoff-gated latent MLP update.
        mlp_out = self.latent_mlp(
            torch.cat([scalar_features, invariants], dim=-1)
        )                                                             # (E, F_s)
        mlp_out = mlp_out * edge_cutoff.unsqueeze(-1)

        # (7) α-residual.
        updated_scalars = (
            self.residual_a * scalar_features + self.residual_b * mlp_out
        )

        # (8) No post-TP tensor rescale — paper-faithful.
        return updated_scalars, new_tensor


# ===========================================================================
# Allegro — encoder
# ===========================================================================


class AllegroSpec(BaseModel):
    """Configuration for the Allegro encoder."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_elements: int = Field(..., gt=0)
    num_scalar_features: int = Field(64, gt=0)
    num_tensor_features: int = Field(16, gt=0)
    r_max: float = Field(..., gt=0.0)
    num_bessel: int = Field(8, gt=0)
    l_max: int = Field(2, ge=0)
    num_layers: int = Field(2, gt=0)
    poly_p: int = Field(6, ge=1)
    scalar_mlp_hiddens: list[int] | None = None
    latent_mlp_hiddens: list[int] | None = None
    avg_num_neighbors: float | None = None
    residual_alpha: float = 0.5


class Allegro(TensorDictModuleBase):
    """Allegro equivariant edge encoder.

    Reads ``(atoms.Z, edges.edge_index, edges.bond_diff, edges.bond_dist)``
    from a :class:`~molix.data.types.GraphBatch` and writes
    ``edges.edge_features`` of shape ``(n_edges, num_layers, num_scalar)``.

    The stored per-layer scalars are already cutoff-gated (the gate is applied
    inside each :class:`AllegroLayer` on the MLP update), so downstream heads
    can consume them directly without re-multiplying by ``u(r_ij)``.

    Reference:
        Musaelian et al. "Learning Local Equivariant Representations for
        Large-Scale Atomistic Dynamics" Nature Communications 14, 579 (2023)
        https://arxiv.org/abs/2204.05249
    """

    in_keys = [
        ("atoms", "Z"),
        ("edges", "edge_index"),
        ("edges", "bond_diff"),
        ("edges", "bond_dist"),
    ]
    out_keys = [("edges", "edge_features")]

    def __init__(
        self,
        *,
        num_elements: int,
        num_scalar_features: int = 64,
        num_tensor_features: int = 16,
        r_max: float,
        num_bessel: int = 8,
        l_max: int = 2,
        num_layers: int = 2,
        poly_p: int = 6,
        scalar_mlp_hiddens: list[int] | None = None,
        latent_mlp_hiddens: list[int] | None = None,
        latent_activation: type[nn.Module] | None = nn.SiLU,
        avg_num_neighbors: float | None = None,
        residual_alpha: float = 0.5,
    ):
        super().__init__()

        self.config = AllegroSpec(
            num_elements=num_elements,
            num_scalar_features=num_scalar_features,
            num_tensor_features=num_tensor_features,
            r_max=r_max,
            num_bessel=num_bessel,
            l_max=l_max,
            num_layers=num_layers,
            poly_p=poly_p,
            scalar_mlp_hiddens=scalar_mlp_hiddens,
            latent_mlp_hiddens=latent_mlp_hiddens,
            avg_num_neighbors=avg_num_neighbors,
            residual_alpha=residual_alpha,
        )

        self.embedding = PairEmbedding(
            num_elements=num_elements,
            num_scalar_features=num_scalar_features,
            num_tensor_features=num_tensor_features,
            r_max=r_max,
            num_bessel=num_bessel,
            l_max=l_max,
            scalar_mlp_hiddens=scalar_mlp_hiddens,
            poly_p=poly_p,
        )

        self.layers = nn.ModuleList(
            [
                AllegroLayer(
                    num_scalar_features=num_scalar_features,
                    num_tensor_features=num_tensor_features,
                    l_max=l_max,
                    latent_mlp_hiddens=latent_mlp_hiddens,
                    latent_activation=latent_activation,
                    avg_num_neighbors=avg_num_neighbors,
                    residual_alpha=residual_alpha,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, td: GraphBatch) -> GraphBatch:
        """Populate ``edges.edge_features`` on ``td`` in place.

        Args:
            td: ``GraphBatch`` with ``atoms`` and ``edges`` sub-dicts.

        Returns:
            The same ``GraphBatch`` with ``edges.edge_features`` of shape
            ``(n_edges, num_layers, num_scalar_features)``.
        """
        Z = td["atoms", "Z"]
        bond_dist = td["edges", "bond_dist"]
        bond_diff = td["edges", "bond_diff"]
        edge_index = td["edges", "edge_index"]
        n_nodes: int = int(Z.shape[0])

        scalar, tensor, edge_angular, edge_cutoff = self.embedding(
            Z=Z,
            bond_dist=bond_dist,
            bond_diff=bond_diff,
            edge_index=edge_index,
        )

        per_layer: list[torch.Tensor] = []
        for layer in self.layers:
            scalar, tensor = layer(
                scalar_features=scalar,
                tensor_features=tensor,
                edge_angular=edge_angular,
                edge_cutoff=edge_cutoff,
                edge_index=edge_index,
                n_nodes=n_nodes,
            )
            per_layer.append(scalar)

        td["edges", "edge_features"] = torch.stack(per_layer, dim=1)
        return td
