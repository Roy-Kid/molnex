"""Numerical parity and invariance tests for Allegro.

Complements ``test_allegro.py`` with two classes of checks:

1. **Parity against a naive reference** — every AllegroLayer / Allegro
   forward is reimplemented in this file in an obviously-correct way
   (explicit Python loops for aggregation and per-channel scaling, no
   cuequivariance-specific orchestration).  The naive version *shares the
   same submodules* as the layer under test (``env_embed``, ``tp``,
   ``tp_linear``, ``latent_mlp``, ``tensor_env``), so any divergence comes
   from the orchestration logic — exactly the code we wrote by hand.

2. **Physical invariants** — cutoff vanishing, permutation invariance of
   neighbors, and translational invariance.  These catch bugs the parity
   test cannot (e.g. a wrongly-normalised aggregation that's self-consistent
   but physically wrong).
"""

from __future__ import annotations

import math

import pytest
import torch

from molix.data.types import AtomData, EdgeData, GraphBatch
from molrep.interaction.tensor_product import EquivariantPolynomialTP
from molzoo.allegro import Allegro, AllegroLayer, PairEmbedding, allegro_uuu_descriptor


# ===========================================================================
# Helpers
# ===========================================================================


def _build_batch(pos: torch.Tensor, Z: torch.Tensor, edge_src_dst: torch.Tensor) -> GraphBatch:
    """Build a GraphBatch given positions, atomic numbers, and edges.

    ``edge_src_dst`` has shape ``(n_edges, 2)``; column 0 is source, column 1
    is destination — consistent with the convention used throughout molzoo.
    """
    n_nodes = pos.shape[0]
    n_edges = edge_src_dst.shape[0]
    bond_diff = pos[edge_src_dst[:, 1]] - pos[edge_src_dst[:, 0]]
    bond_dist = bond_diff.norm(dim=-1).clamp(min=1e-4)
    return GraphBatch(
        atoms=AtomData(
            Z=Z,
            pos=pos,
            batch=torch.zeros(n_nodes, dtype=torch.long),
            batch_size=[n_nodes],
        ),
        edges=EdgeData(
            edge_index=edge_src_dst,
            bond_diff=bond_diff,
            bond_dist=bond_dist,
            batch_size=[n_edges],
        ),
        batch_size=[],
    )


def _naive_encoder_aggregate(
    edge_angular: torch.Tensor,
    edge_cutoff: torch.Tensor,
    edge_index: torch.Tensor,
    n_nodes: int,
    avg_num_neighbors: float | None,
) -> torch.Tensor:
    """Python-loop reimplementation of the encoder's neighbour aggregation.

    Matches the vectorised ``scatter_add_`` in ``Allegro.forward`` by walking
    edges one at a time, so a bug in the encoder's fused kernel (wrong src
    column, dropped cutoff weight, missing normalisation) shows up as a
    parity mismatch rather than a subtle numerical drift.
    """
    sh_dim = edge_angular.shape[1]
    node_Y = torch.zeros(
        n_nodes, sh_dim, dtype=edge_angular.dtype, device=edge_angular.device
    )
    cutoff_counts = torch.zeros(
        n_nodes, dtype=edge_angular.dtype, device=edge_angular.device
    )
    for e in range(edge_index.shape[0]):
        i = int(edge_index[e, 0])
        node_Y[i] = node_Y[i] + edge_cutoff[e] * edge_angular[e]
        cutoff_counts[i] = cutoff_counts[i] + edge_cutoff[e]

    if avg_num_neighbors is not None:
        return node_Y / math.sqrt(avg_num_neighbors)
    return node_Y / cutoff_counts.clamp(min=1.0).sqrt().unsqueeze(-1)


def _naive_encoder_forward(
    encoder: Allegro,
    graph: GraphBatch,
) -> torch.Tensor:
    """Python-loop reference for ``Allegro.forward``.

    Shares every submodule with the encoder under test; only the
    aggregation + layer-loop orchestration is re-derived naively, so any
    divergence points at the fused ``scatter_add_`` / gather logic.
    """
    Z = graph["atoms", "Z"]
    bond_dist = graph["edges", "bond_dist"]
    bond_diff = graph["edges", "bond_diff"]
    edge_index = graph["edges", "edge_index"]
    n_nodes = int(Z.shape[0])

    scalar, tensor, edge_angular, edge_cutoff = encoder.embedding(
        Z=Z, bond_dist=bond_dist, bond_diff=bond_diff, edge_index=edge_index
    )

    node_Y = _naive_encoder_aggregate(
        edge_angular, edge_cutoff, edge_index, n_nodes, encoder.avg_num_neighbors
    )
    src = edge_index[:, 0]
    edge_node_Y = torch.stack([node_Y[int(s)] for s in src])

    for layer in encoder.layers:
        scalar, tensor = layer(
            scalar_features=scalar,
            tensor_features=tensor,
            edge_node_Y=edge_node_Y,
            edge_cutoff=edge_cutoff,
        )
    return scalar * edge_cutoff.unsqueeze(-1)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def graph_batch():
    """Small chain of 4 atoms with 6 directed edges."""
    torch.manual_seed(0)
    edge_index = torch.tensor(
        [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=torch.long
    )
    pos = torch.randn(4, 3)
    Z = torch.randint(0, 5, (4,))
    return _build_batch(pos, Z, edge_index)


# ===========================================================================
# Parity tests — encoder
# ===========================================================================


def _make_encoder(avg_num_neighbors: float | None, seed: int) -> Allegro:
    torch.manual_seed(seed)
    return Allegro(
        num_elements=5,
        num_scalar_features=16,
        num_tensor_features=8,
        r_max=5.0,
        l_max=2,
        num_layers=2,
        avg_num_neighbors=avg_num_neighbors,
    ).eval()


class TestAllegroEncoderParity:
    """Parity between ``Allegro.forward`` and a naive loop-based reference."""

    def test_parity_forward_avg_num_neighbors_none(self, graph_batch):
        encoder = _make_encoder(avg_num_neighbors=None, seed=2)
        with torch.no_grad():
            actual = encoder(graph_batch)["edges", "edge_features"]
            ref = _naive_encoder_forward(encoder, graph_batch)
        assert torch.allclose(actual, ref, rtol=1e-5, atol=1e-5), (
            f"edge feature mismatch: max diff {(actual - ref).abs().max().item():.3e}"
        )

    def test_parity_forward_avg_num_neighbors_set(self, graph_batch):
        encoder = _make_encoder(avg_num_neighbors=4.0, seed=3)
        with torch.no_grad():
            actual = encoder(graph_batch)["edges", "edge_features"]
            ref = _naive_encoder_forward(encoder, graph_batch)
        assert torch.allclose(actual, ref, rtol=1e-5, atol=1e-5)

    def test_parity_backward(self, graph_batch):
        """Gradients through the encoder match the naive reference."""
        encoder = _make_encoder(avg_num_neighbors=4.0, seed=4)

        pos_a = graph_batch["atoms", "pos"].clone().detach().requires_grad_(True)
        graph_a = _build_batch(
            pos_a, graph_batch["atoms", "Z"], graph_batch["edges", "edge_index"]
        )
        actual = encoder(graph_a)["edges", "edge_features"]
        actual.sum().backward()
        grad_actual = pos_a.grad.clone()

        pos_r = graph_batch["atoms", "pos"].clone().detach().requires_grad_(True)
        graph_r = _build_batch(
            pos_r, graph_batch["atoms", "Z"], graph_batch["edges", "edge_index"]
        )
        ref = _naive_encoder_forward(encoder, graph_r)
        ref.sum().backward()
        grad_ref = pos_r.grad.clone()

        assert torch.allclose(grad_actual, grad_ref, rtol=1e-4, atol=1e-5), (
            f"pos grad mismatch: {(grad_actual - grad_ref).abs().max().item():.3e}"
        )


# ===========================================================================
# Physical invariants — full encoder
# ===========================================================================


class TestAllegroInvariants:
    """End-to-end physical invariants of the full Allegro encoder."""

    def test_neighbor_permutation_invariance(self):
        """Permuting edges in edge_index shouldn't change per-edge outputs.

        The aggregation is a sum over neighbors — by commutativity it must
        be invariant to the order in which neighbors appear in edge_index.
        """
        torch.manual_seed(5)
        pos = torch.randn(4, 3)
        Z = torch.randint(0, 5, (4,))
        edge_index = torch.tensor(
            [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=torch.long
        )

        encoder = Allegro(
            num_elements=5,
            num_scalar_features=16,
            num_tensor_features=8,
            r_max=5.0,
            l_max=2,
            num_layers=2,
        ).eval()

        g_ordered = _build_batch(pos, Z, edge_index)
        with torch.no_grad():
            out_ordered = encoder(g_ordered)["edges", "edge_features"]

        # Permute edges (the aggregation is a sum, so same output — just reordered)
        perm = torch.tensor([3, 0, 5, 1, 2, 4], dtype=torch.long)
        g_perm = _build_batch(pos, Z, edge_index[perm])
        with torch.no_grad():
            out_perm = encoder(g_perm)["edges", "edge_features"]

        # Align: out_perm[e] corresponds to original edge perm[e]
        out_perm_aligned = torch.empty_like(out_ordered)
        out_perm_aligned[perm] = out_perm
        assert torch.allclose(out_ordered, out_perm_aligned, rtol=1e-5, atol=1e-5), (
            f"max diff {(out_ordered - out_perm_aligned).abs().max().item():.3e}"
        )

    def test_cutoff_vanishing_of_output(self):
        """Edge features of an out-of-cutoff edge vanish.

        Place an edge whose bond distance is > ``r_max``. Its polynomial
        cutoff u(r) is 0, so both (a) its contribution to neighbour aggregates
        and (b) its own output features should be zero.
        """
        torch.manual_seed(6)
        r_max = 3.0

        # Two isolated pairs. Pair (0, 1) at distance 1.0; pair (2, 3) at 5.0 > r_max.
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0], [15.0, 0.0, 0.0]]
        )
        Z = torch.tensor([1, 1, 1, 1], dtype=torch.long)
        edge_index = torch.tensor(
            [[0, 1], [1, 0], [2, 3], [3, 2]], dtype=torch.long
        )

        encoder = Allegro(
            num_elements=5,
            num_scalar_features=16,
            num_tensor_features=8,
            r_max=r_max,
            l_max=2,
            num_layers=2,
        ).eval()
        g = _build_batch(pos, Z, edge_index)
        with torch.no_grad():
            features = encoder(g)["edges", "edge_features"]

        # Edges 2 and 3 are out of cutoff → should be zero
        assert features[2].abs().max() < 1e-6, (
            f"out-of-cutoff edge 2 leaked: {features[2].abs().max().item():.3e}"
        )
        assert features[3].abs().max() < 1e-6, (
            f"out-of-cutoff edge 3 leaked: {features[3].abs().max().item():.3e}"
        )
        # In-cutoff edges should be non-zero (sanity check the test is meaningful)
        assert features[0].abs().max() > 1e-4
        assert features[1].abs().max() > 1e-4

    def test_out_of_cutoff_neighbor_does_not_contaminate(self):
        """A neighbour beyond r_cut must not affect in-cutoff edge features.

        Graph A: pair (0, 1) within r_cut.
        Graph B: same pair plus atom 2 outside r_cut of both.
        Feature of edge (0, 1) must be identical between A and B.
        """
        torch.manual_seed(7)
        r_max = 3.0

        # Graph A: single in-cutoff pair
        pos_a = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        Z_a = torch.tensor([1, 1], dtype=torch.long)
        edge_a = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        g_a = _build_batch(pos_a, Z_a, edge_a)

        # Graph B: add atom 2 far away; atoms 0 and 2 not connected (beyond r_cut).
        # We DO include edges (1, 2) and (2, 1) past r_cut to test that they are
        # masked by the cutoff inside the aggregation.
        pos_b = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
        )
        Z_b = torch.tensor([1, 1, 1], dtype=torch.long)
        edge_b = torch.tensor(
            [[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long
        )
        g_b = _build_batch(pos_b, Z_b, edge_b)

        encoder = Allegro(
            num_elements=5,
            num_scalar_features=16,
            num_tensor_features=8,
            r_max=r_max,
            l_max=2,
            num_layers=2,
        ).eval()
        with torch.no_grad():
            f_a = encoder(g_a)["edges", "edge_features"]
            f_b = encoder(g_b)["edges", "edge_features"]

        # Edges (0, 1) and (1, 0) correspond to indices 0 and 1 in both graphs.
        diff = (f_a[:2] - f_b[:2]).abs().max().item()
        assert diff < 1e-5, (
            f"out-of-cutoff neighbour contaminated in-cutoff edge: max diff {diff:.3e}"
        )

    def test_translation_invariance(self):
        """Shifting all positions by a constant leaves outputs unchanged."""
        torch.manual_seed(8)
        pos = torch.randn(4, 3)
        Z = torch.randint(0, 5, (4,))
        edge_index = torch.tensor(
            [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=torch.long
        )

        encoder = Allegro(
            num_elements=5,
            num_scalar_features=16,
            num_tensor_features=8,
            r_max=5.0,
            l_max=2,
            num_layers=2,
        ).eval()
        g1 = _build_batch(pos, Z, edge_index)
        g2 = _build_batch(pos + torch.tensor([7.3, -2.1, 0.5]), Z, edge_index)
        with torch.no_grad():
            f1 = encoder(g1)["edges", "edge_features"]
            f2 = encoder(g2)["edges", "edge_features"]
        assert torch.allclose(f1, f2, rtol=1e-5, atol=1e-5)


# ===========================================================================
# Allegro per-channel ("uuu") descriptor
# ===========================================================================


class TestAllegroUUUDescriptor:
    """Descriptor for the reference per-channel Allegro tensor product."""

    def test_subscripts_and_dims(self):
        """Builder produces ``"u,iu,ju,ku+ijk"`` with matched multiplicity."""
        import cuequivariance as cue

        irreps_in = cue.Irreps("O3", "8x0e + 8x1o + 8x2e")
        irreps_sh = cue.Irreps("O3", "1x0e + 1x1o + 1x2e")
        poly = allegro_uuu_descriptor(irreps_in, irreps_sh)

        ops = poly.polynomial.operations
        assert len(ops) == 1
        stp = ops[0][1]
        assert stp.subscripts == "u,iu,ju,ku+ijk"
        assert poly.inputs[0].irreps.dim == stp.operands[0].size
        for mul, _ir in poly.outputs[0].irreps:
            assert mul == irreps_in.muls[0]

    def test_rejects_non_uniform_mul(self):
        """Non-uniform multiplicity is not supported by the uuu subscripts."""
        import cuequivariance as cue

        with pytest.raises(ValueError, match="uniform multiplicity"):
            allegro_uuu_descriptor(
                cue.Irreps("O3", "8x0e + 16x1o"),
                cue.Irreps("O3", "1x0e + 1x1o"),
            )

    def test_forward_shape_via_wrapper(self):
        """Dispatched through ``EquivariantPolynomialTP`` with naive method."""
        import cuequivariance as cue

        num_tensor = 8
        irreps_in = cue.Irreps(
            "O3", f"{num_tensor}x0e + {num_tensor}x1o + {num_tensor}x2e"
        )
        irreps_sh = cue.Irreps("O3", "1x0e + 1x1o + 1x2e")
        poly = allegro_uuu_descriptor(irreps_in, irreps_sh)
        tp = EquivariantPolynomialTP(
            poly, shared_weights=False, internal_weights=False, method="naive"
        )

        n_edges = 4
        lhs = torch.randn(n_edges, irreps_in.dim)
        rhs = torch.randn(n_edges, poly.inputs[2].irreps.dim)
        w = torch.randn(n_edges, tp.weight_numel)
        out = tp(lhs, rhs, weight=w)

        assert out.shape == (n_edges, tp.irreps_out.dim)
        for mul, _ir in tp.irreps_out:
            assert mul == num_tensor
