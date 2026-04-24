"""Allegro: local equivariant edge encoder.

Pair-level equivariant encoder using iterated tensor products with neighbourhood
aggregation. Returns per-layer per-edge scalar features only — readout, energy
aggregation, and any rescaling live in :mod:`molpot.heads`.

Cutoff handling follows the paper + common practice: (a) the polynomial
``u(r_ij)`` weights each edge inside the neighbour aggregation so
out-of-cutoff neighbours drop out cleanly; (b) ``u(r_ij)`` gates every
layer's MLP update so scalar growth is smooth near the cutoff; and
(c) a final ``u(r_ij)`` multiplier on the encoder output forces the
edge feature to zero at ``r = r_cut`` (needed for energy and force
continuity — without it the α-residual leaves an ``a^L · scalar_0``
floor on out-of-cutoff edges).

See ``src/molzoo/specs/allegro.md`` for the full math.

Reference:
    Musaelian et al. "Learning Local Equivariant Representations for
    Large-Scale Atomistic Dynamics" Nature Communications 14, 579 (2023)
    https://arxiv.org/abs/2204.05249
"""

from __future__ import annotations

import itertools
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
# Model-specific descriptor (per-channel Allegro TP, "u,iu,ju,ku+ijk")
# ===========================================================================


def allegro_uuu_descriptor(
    irreps_in: cue.Irreps,
    irreps_sh: cue.Irreps,
) -> cue.EquivariantPolynomial:
    r"""Per-channel Allegro tensor product descriptor.

    Subscripts: ``weights[u], lhs[iu], rhs[ju], output[ku]``.

    This is the descriptor used by the reference MIR Allegro ``Contracter``
    kernel:

    .. math::
       \text{out}_{k,u} = \sum_{i,j} \mathrm{CG}_{i,j,k}\, w_u\, V_{i,u}\, v_{j,u}

    Unlike :func:`cue.descriptors.channelwise_tensor_product` (subscripts
    ``"uv,iu,jv,kuv+ijk"``), here both input operands share the **same**
    channel dimension ``u``, and there is one scalar weight per channel per
    CG path. This keeps the output multiplicity at ``u`` (no ``u*v``
    blowup) while preserving per-channel, per-path mixing — what Allegro
    requires.

    Args:
        irreps_in: Irreps of the lhs and output (they share multiplicity
            ``u``). Typically ``num_tensor * (0e + 1o + 2e + ...)``. Must
            have uniform mul.
        irreps_sh: Irreps of the spherical harmonics. The caller is
            responsible for broadcasting the single-channel ``Y_ij`` to
            ``u`` channels before passing to the executing module.

    Returns:
        :class:`cue.EquivariantPolynomial` with inputs
        ``(weights, lhs, rhs)`` and one output.
    """
    if len(set(irreps_in.muls)) != 1:
        raise ValueError(
            "allegro_uuu_descriptor requires irreps_in to have uniform "
            f"multiplicity (got muls={irreps_in.muls})"
        )
    u = irreps_in.muls[0]
    G = irreps_in.irrep_class
    d = cue.SegmentedTensorProduct.from_subscripts("u,iu,ju,ku+ijk")

    for _mul, ir in irreps_in:
        d.add_segment(1, (ir.dim, u))
    for _mul, ir in irreps_sh:
        d.add_segment(2, (ir.dim, u))

    irreps_out_list: list[tuple[int, cue.Irrep]] = []
    for (i1, (_m1, ir1)), (i2, (_m2, ir2)) in itertools.product(
        enumerate(irreps_in), enumerate(irreps_sh)
    ):
        for ir3 in ir1 * ir2:
            for cg in G.clebsch_gordan(ir1, ir2, ir3):
                d.add_path(None, i1, i2, None, c=cg, dims={"u": u})
                irreps_out_list.append((u, ir3))

    irreps_out = cue.Irreps(G, irreps_out_list)
    irreps_out, _perm, inv = irreps_out.sort()
    d = d.permute_segments(3, inv)
    d = d.normalize_paths_for_operand(-1)

    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(
                irreps_in.new_scalars(d.operands[0].size), cue.ir_mul
            ),
            cue.IrrepsAndLayout(irreps_in, cue.ir_mul),
            cue.IrrepsAndLayout(
                cue.Irreps(G, [(u, ir) for _, ir in irreps_sh]), cue.ir_mul
            ),
        ],
        [cue.IrrepsAndLayout(irreps_out, cue.ir_mul)],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )


# ===========================================================================
# Helpers — pure functions on ir_mul-layout tensors
# ===========================================================================
#
# Both helpers assume **uniform multiplicity** across all irreps (the Allegro
# convention: ``u`` channels per l). Under that constraint an ir_mul-layout
# tensor of ``sum_l (2l+1) * u`` flat dims can be reshaped as
# ``(batch, sh_dim, u)`` — the u (channel) axis stays contiguous and the
# sh_dim (m-component) axis aggregates all l blocks in order. That lets us
# replace a per-l Python loop with a single reshape + broadcast.


def _env_weight_harmonics(
    edge_angular: torch.Tensor,
    env_weights: torch.Tensor,
    num_tensor_features: int,
) -> torch.Tensor:
    """Weight spherical harmonics by per-channel environment weights.

    Computes
    ``result[e, m, c] = edge_angular[e, m] * env_weights[e, c]`` and lays it
    out in ``ir_mul`` order (m-major, channel-fast) — a single outer-product
    + reshape, with no per-l loop.

    Args:
        edge_angular: Spherical harmonics ``(n_edges, sh_dim)`` (mul=1).
        env_weights: Per-channel weights ``(n_edges, num_tensor_features)``.
        num_tensor_features: Number of tensor feature channels (the uniform
            multiplicity ``u`` used throughout the layer).

    Returns:
        Weighted tensor features ``(n_edges, sh_dim * num_tensor_features)``
        in ir_mul layout.
    """
    n_edges, sh_dim = edge_angular.shape
    return (edge_angular.unsqueeze(-1) * env_weights.unsqueeze(-2)).reshape(
        n_edges, sh_dim * num_tensor_features
    )


def _scale_by_channel(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    num_tensor_features: int,
) -> torch.Tensor:
    """Scale an ir_mul tensor feature by per-channel scalar weights.

    Args:
        tensor: Equivariant tensor in ir_mul layout
            ``(n_edges, sh_dim * num_tensor_features)``. ``sh_dim`` is the
            total number of m-components across all l.
        scale: Per-channel scale factors ``(n_edges, num_tensor_features)``.
        num_tensor_features: Uniform channel multiplicity ``u``.

    Returns:
        Scaled tensor with the same shape as ``tensor``.
    """
    n_edges, flat_dim = tensor.shape
    sh_dim = flat_dim // num_tensor_features
    return (
        tensor.view(n_edges, sh_dim, num_tensor_features) * scale.unsqueeze(1)
    ).reshape(n_edges, flat_dim)


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
            edge_angular, env_weights, self.num_tensor_features
        )

        return scalar_features, tensor_features, edge_angular, edge_cutoff


# ===========================================================================
# AllegroLayer — single TP-based layer with cutoff-gated residual
# ===========================================================================


class AllegroLayer(nn.Module):
    """Pair-level tensor product layer with cutoff-gated residual update.

    Flow, for each edge ``(i, j)``, given a pre-aggregated neighbourhood
    ``edge_node_Y_ij = Y_i[source(ij)]`` supplied by the encoder:

    1. ``env_w_ij = Linear(scalar_ij)`` → per-channel env weight ``(E, u)``.
    2. ``V_scaled_ij = env_w_ij · V_ij`` (channel-wise scale).
    3. ``new_tensor_ij = Linear(TP(V_scaled_ij, edge_node_Y_ij))``  (paper eq. 21).
    4. ``invariants_ij = new_tensor_ij[:, :u]``           (L=0 scalars).
    5. ``mlp_out_ij = u(r_ij) · LatentMLP([scalar_ij, invariants_ij])``
       — **the cutoff gate is applied per-layer on the MLP output, matching
       ``mir-group/allegro::_allegro.py:494``**.
    6. ``scalar_ij ← a · scalar_ij + b · mlp_out_ij`` with
       ``a = 1/√(1+α²), b = α/√(1+α²)``  (paper SI α-residual, α=0.5).
    7. ``V_ij ← new_tensor_ij``  (no post-TP rescale — matches paper).

    The neighbour aggregation ``Y_i = Σ_{k∈N(i)} Y_ik`` and its
    ``1/√avg_num_neighbors`` normalisation are layer-invariant (they depend
    only on connectivity and the spherical harmonics, both fixed across the
    layer stack). The encoder computes them once and gathers them per-edge
    via ``Y_i[source(ij)]`` before the layer loop — saving ``(L-1)``
    scatter_adds per forward.

    The single-channel right operand in step 3 lets us use the fast
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
        residual_alpha: float = 0.5,
    ):
        super().__init__()

        self.num_scalar_features = num_scalar_features
        self.num_tensor_features = num_tensor_features
        self.l_max = l_max

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

    def forward(
        self,
        scalar_features: torch.Tensor,
        tensor_features: torch.Tensor,
        edge_node_Y: torch.Tensor,
        edge_cutoff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one Allegro layer.

        Args:
            scalar_features: ``(n_edges, F_s)`` current scalar track.
            tensor_features: ``(n_edges, irreps_dim)`` current tensor track.
            edge_node_Y: ``(n_edges, sh_dim)`` pre-aggregated neighbourhood
                spherical harmonics gathered per-edge from the source node
                (computed once by the encoder; see :class:`Allegro.forward`).
            edge_cutoff: ``(n_edges,)`` polynomial-cutoff factor ``u(r_ij)``.

        Returns:
            ``(scalar_out, tensor_out)`` with the same shapes as the inputs.
            The per-layer MLP update is gated by ``u(r_ij)`` so scalar
            growth is smooth; the encoder additionally multiplies the
            final scalar by ``u(r_ij)`` so out-of-cutoff edges produce
            literal zero features.
        """
        # (1-2) Pre-scale V by env weights.
        env_w = self.env_embed(scalar_features)                       # (E, u)
        scaled_V = _scale_by_channel(
            tensor_features, env_w, self.num_tensor_features
        )                                                             # (E, irreps)

        # (3) Tensor product → project back to input irreps.
        tp_out = self.tp(scaled_V, edge_node_Y)
        new_tensor = self.tp_linear(tp_out)                           # (E, irreps)

        # (4) Extract L=0 invariants.
        invariants = new_tensor[:, : self.num_tensor_features]        # (E, u)

        # (5) Cutoff-gated latent MLP update.
        mlp_out = self.latent_mlp(
            torch.cat([scalar_features, invariants], dim=-1)
        )                                                             # (E, F_s)
        mlp_out = mlp_out * edge_cutoff.unsqueeze(-1)

        # (6) α-residual.
        updated_scalars = (
            self.residual_a * scalar_features + self.residual_b * mlp_out
        )

        # (7) No post-TP tensor rescale — paper-faithful.
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
    from a :class:`~molix.data.types.GraphBatch` and writes the final layer's
    per-edge scalar track to ``edges.edge_features`` with shape
    ``(n_edges, num_scalar_features)``.

    The output scalar is cutoff-gated at two points: every :class:`AllegroLayer`
    multiplies its MLP update by ``u(r_ij)`` (smooth residual growth), and the
    encoder multiplies the final scalar by ``u(r_ij)`` one more time so
    out-of-cutoff edges produce zero output — downstream heads can consume it
    directly without re-multiplying by ``u(r_ij)``.

    Only the last layer is stored: intermediate layers are transient updates to
    the same ``(E, F)`` tensor and the paper's ``edge_eng`` readout consumes
    only the final scalar. Stacking every layer would allocate ``num_layers ×``
    memory for outputs that are immediately discarded.

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

        self.avg_num_neighbors = (
            float(avg_num_neighbors) if avg_num_neighbors is not None else None
        )
        with cue.assume(O3):
            self._sh_dim = Irreps(sh_irreps_from_l_max(l_max)).dim

        self.layers = nn.ModuleList(
            [
                AllegroLayer(
                    num_scalar_features=num_scalar_features,
                    num_tensor_features=num_tensor_features,
                    l_max=l_max,
                    latent_mlp_hiddens=latent_mlp_hiddens,
                    latent_activation=latent_activation,
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
            ``(n_edges, num_scalar_features)`` — the final layer's scalar
            track.
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

        # Neighbour aggregation Y_i = Σ_{k∈N(i)} u(r_ik) Y_ik. The cutoff
        # factor u(r_ik) inside the sum is what makes out-of-cutoff edges
        # drop out — without it, a neighbour past r_max still contributes
        # its Y to node_Y[i] and contaminates i's in-cutoff edges.
        # edge_angular and edge_index are fixed across the layer stack, so
        # we aggregate once here and gather per-edge; the same math happens
        # inside every layer in the naive reference.
        src = edge_index[:, 0]
        weighted_Y = edge_angular * edge_cutoff.unsqueeze(-1)
        node_Y = torch.zeros(
            n_nodes,
            self._sh_dim,
            dtype=edge_angular.dtype,
            device=edge_angular.device,
        )
        node_Y.scatter_add_(
            0, src.unsqueeze(-1).expand_as(weighted_Y), weighted_Y
        )
        if self.avg_num_neighbors is not None:
            node_Y = node_Y / math.sqrt(self.avg_num_neighbors)
        else:
            # Normalise by cutoff-weighted count so the denominator reflects
            # the same "effective neighbour mass" that the numerator carries.
            src_count = torch.zeros(
                n_nodes, dtype=edge_angular.dtype, device=edge_angular.device
            )
            src_count.scatter_add_(0, src, edge_cutoff)
            node_Y = node_Y / src_count.clamp(min=1.0).sqrt().unsqueeze(-1)
        edge_node_Y = node_Y[src]                                     # (E, sh_dim)

        for layer in self.layers:
            scalar, tensor = layer(
                scalar_features=scalar,
                tensor_features=tensor,
                edge_node_Y=edge_node_Y,
                edge_cutoff=edge_cutoff,
            )

        # Gate the final output by u(r_ij) so the energy/force surface is
        # continuous at r_cut: the layer's α-residual would otherwise leave
        # an ``a^L · scalar_0`` floor on out-of-cutoff edges, breaking
        # energy conservation near the cutoff.
        td["edges", "edge_features"] = scalar * edge_cutoff.unsqueeze(-1)
        return td
