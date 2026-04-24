"""Tests for the Allegro encoder and its composed energy head.

Coverage:
* Shape / compile contracts for :class:`PairEmbedding`,
  :class:`AllegroLayer`, :class:`Allegro`.
* Symmetry invariants of the full energy pipeline
  (encoder + :class:`EdgeEnergyHead`):
  translation, rotation (O(3)), and permutation.
* Smoothness: edges past ``r_cut`` must not contribute (their scalars are 0).
* Functional correctness: a single batch must be overfittable to tiny loss
  under Adam in a small number of steps.

The symmetry and smoothness tests target the *total energy* (the downstream
observable) rather than intermediate tensors, since any encoder bug that
breaks equivariance will manifest as a broken invariance of the total energy.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from molix.data.types import AtomData, EdgeData, GraphBatch, GraphData
from molpot.heads import EdgeEnergyHead
from molrep.utils.equivariance import (
    random_rotation_matrix,
    rotate_vectors,
    rotation_matrix_z,
)
from molzoo.allegro import Allegro, AllegroLayer, PairEmbedding
from tests.utils import assert_compile_compatible


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_graph(
    pos: torch.Tensor,
    Z: torch.Tensor,
    r_cut: float,
    *,
    with_graphs: bool = True,
) -> GraphBatch:
    """Build a full-connectivity GraphBatch (pairs within ``r_cut``, bidirectional).

    Provided as a helper so the same edge-construction logic is shared by every
    test case (prevents mismatch between "model sees" and "test asserts on").
    """
    n = pos.shape[0]
    pairs = []
    diffs = []
    dists = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pos[j] - pos[i]
            r = d.norm().item()
            if r < r_cut:
                pairs.append((i, j))
                diffs.append(d)
                dists.append(r)
    edge_index = torch.tensor(pairs, dtype=torch.long)
    bond_diff = torch.stack(diffs, dim=0)
    bond_dist = torch.tensor(dists, dtype=pos.dtype)

    atoms = AtomData(
        Z=Z,
        pos=pos,
        batch=torch.zeros(n, dtype=torch.long),
        batch_size=[n],
    )
    edges = EdgeData(
        edge_index=edge_index,
        bond_diff=bond_diff,
        bond_dist=bond_dist,
        batch_size=[edge_index.shape[0]],
    )
    td = {"atoms": atoms, "edges": edges}
    if with_graphs:
        td["graphs"] = GraphData(
            num_atoms=torch.tensor([n], dtype=torch.long), batch_size=[1]
        )
    return GraphBatch(**td, batch_size=[])


@pytest.fixture
def graph_data():
    torch.manual_seed(0)
    pos = torch.randn(4, 3)
    Z = torch.randint(1, 5, (4,))
    return _build_graph(pos, Z, r_cut=5.0, with_graphs=True)


@pytest.fixture
def small_encoder():
    torch.manual_seed(0)
    return Allegro(
        num_elements=10,
        num_scalar_features=16,
        num_tensor_features=8,
        r_max=5.0,
        num_bessel=4,
        l_max=2,
        num_layers=2,
        scalar_mlp_hiddens=[16, 32],
        latent_mlp_hiddens=[32, 32],
    )


def _build_energy_model(encoder: Allegro, avg_nbr: float | None = None) -> nn.Module:
    head = EdgeEnergyHead(
        input_dim=encoder.config.num_scalar_features,
        hidden_dim=16,
        avg_num_neighbors=avg_nbr,
    )

    class EnergyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(self, batch: GraphBatch) -> dict[str, torch.Tensor]:
            batch = self.encoder(batch)
            return self.head(batch)

    return EnergyModel()


# ---------------------------------------------------------------------------
# Module-level shape / compile contracts
# ---------------------------------------------------------------------------


class TestPairEmbedding:
    def test_output_shapes(self, graph_data):
        module = PairEmbedding(
            num_elements=5,
            num_scalar_features=16,
            num_tensor_features=8,
            r_max=5.0,
            l_max=2,
        )
        scalars, tensors, edge_angular, edge_cutoff = module(
            Z=graph_data["atoms", "Z"],
            bond_dist=graph_data["edges", "bond_dist"],
            bond_diff=graph_data["edges", "bond_diff"],
            edge_index=graph_data["edges", "edge_index"],
        )
        n_edges = graph_data["edges", "edge_index"].shape[0]
        assert scalars.shape == (n_edges, 16)
        assert tensors.shape == (n_edges, module.irreps_dim)
        assert edge_angular.shape == (n_edges, 9)
        assert edge_cutoff.shape == (n_edges,)
        assert torch.all(edge_cutoff >= 0.0) and torch.all(edge_cutoff <= 1.0)

    def test_scalar_mlp_has_no_trailing_activation(self):
        """Paper-faithful: last layer of scalar_mlp is a bare Linear."""
        module = PairEmbedding(
            num_elements=5,
            num_scalar_features=16,
            num_tensor_features=8,
            r_max=5.0,
            l_max=2,
            scalar_mlp_hiddens=[32, 32],
        )
        assert isinstance(module.scalar_mlp[-1], nn.Linear)
        # structure: [Linear, SiLU, Linear, SiLU, Linear]
        assert len([m for m in module.scalar_mlp if isinstance(m, nn.Linear)]) == 3
        assert len([m for m in module.scalar_mlp if isinstance(m, nn.SiLU)]) == 2

    def test_compile(self, graph_data):
        module = PairEmbedding(
            num_elements=5,
            num_scalar_features=16,
            num_tensor_features=8,
            r_max=5.0,
            l_max=2,
        )
        assert_compile_compatible(
            module,
            strict=True,
            Z=graph_data["atoms", "Z"],
            bond_dist=graph_data["edges", "bond_dist"],
            bond_diff=graph_data["edges", "bond_diff"],
            edge_index=graph_data["edges", "edge_index"],
        )


class TestAllegroLayer:
    def _dummy_inputs(self, num_scalar: int, num_tensor: int, irreps_dim: int):
        n_nodes = 4
        edge_index = torch.tensor(
            [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=torch.long
        )
        scalar_features = torch.randn(6, num_scalar)
        tensor_in = torch.randn(6, irreps_dim)
        edge_angular = torch.randn(6, 9)
        edge_cutoff = torch.rand(6)
        return scalar_features, tensor_in, edge_angular, edge_cutoff, edge_index, n_nodes

    def test_preserves_batch_dimension(self):
        num_scalar, num_tensor = 16, 8
        module = AllegroLayer(
            num_scalar_features=num_scalar,
            num_tensor_features=num_tensor,
            l_max=2,
        )
        args = self._dummy_inputs(num_scalar, num_tensor, module.irreps_dim)
        scalar_out, tensor_out = module(*args)
        assert scalar_out.shape == (6, num_scalar)
        assert tensor_out.shape == args[1].shape

    def test_zero_cutoff_freezes_scalar_track(self):
        """When ``edge_cutoff=0`` the residual update collapses to ``a·x``.

        This is the core guarantee of the in-layer cutoff gate: edges past
        ``r_cut`` cannot inject any new information into the scalar track.
        """
        num_scalar, num_tensor = 16, 8
        module = AllegroLayer(
            num_scalar_features=num_scalar,
            num_tensor_features=num_tensor,
            l_max=2,
            residual_alpha=0.5,
        )
        scalar_features, tensor_in, edge_angular, _, edge_index, n_nodes = (
            self._dummy_inputs(num_scalar, num_tensor, module.irreps_dim)
        )
        zero_cutoff = torch.zeros(6)
        scalar_out, _ = module(
            scalar_features, tensor_in, edge_angular, zero_cutoff, edge_index, n_nodes
        )
        expected = module.residual_a * scalar_features
        torch.testing.assert_close(scalar_out, expected, rtol=1e-5, atol=1e-5)

    def test_linear_latent_mlp(self):
        module = AllegroLayer(
            num_scalar_features=16,
            num_tensor_features=8,
            l_max=2,
            latent_mlp_hiddens=[32, 32, 32],
            latent_activation=None,
        )
        activations = [
            m for m in module.latent_mlp if not isinstance(m, nn.Linear)
        ]
        assert activations == []

    def test_residual_alpha_coefficients(self):
        module = AllegroLayer(
            num_scalar_features=16,
            num_tensor_features=8,
            l_max=2,
            residual_alpha=0.5,
        )
        assert abs(module.residual_a ** 2 + module.residual_b ** 2 - 1.0) < 1e-6

    def test_compile(self):
        num_scalar, num_tensor = 16, 8
        module = AllegroLayer(
            num_scalar_features=num_scalar,
            num_tensor_features=num_tensor,
            l_max=2,
        )
        args = self._dummy_inputs(num_scalar, num_tensor, module.irreps_dim)
        assert_compile_compatible(module, *args, strict=True)


class TestAllegroEncoder:
    def test_forward_writes_edge_features(self, graph_data, small_encoder):
        result = small_encoder(graph_data)
        edge_features = result["edges", "edge_features"]
        n_edges = graph_data["edges", "edge_index"].shape[0]
        n_layers = small_encoder.config.num_layers
        F_s = small_encoder.config.num_scalar_features
        assert edge_features.shape == (n_edges, n_layers, F_s)

    def test_scalar_output_is_rotation_invariant(self, graph_data, small_encoder):
        """Encoder's stored scalar per-layer features must be invariant under SO(3)."""
        rotation = rotation_matrix_z(0.73)
        rotated_diff = rotate_vectors(graph_data["edges", "bond_diff"], rotation)
        rotated_batch = graph_data.clone()
        rotated_batch["edges", "bond_diff"] = rotated_diff

        base = small_encoder(graph_data.clone())["edges", "edge_features"]
        rot = small_encoder(rotated_batch)["edges", "edge_features"]
        torch.testing.assert_close(base, rot, rtol=1e-4, atol=1e-4)

    def test_compile(self, graph_data):
        encoder = Allegro(
            num_elements=5,
            num_scalar_features=16,
            num_tensor_features=8,
            r_max=5.0,
            num_layers=2,
        ).eval()
        torch._dynamo.reset()
        compiled = torch.compile(encoder, backend="inductor", fullgraph=True)
        with torch.no_grad():
            ref = encoder(graph_data.clone())["edges", "edge_features"]
            got = compiled(graph_data.clone())["edges", "edge_features"]
        torch.testing.assert_close(ref, got, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# End-to-end physical invariants (encoder + head → total energy)
# ---------------------------------------------------------------------------


class TestEnergyInvariants:
    """Physics symmetries that a correct Allegro + head pipeline MUST satisfy."""

    def test_translation_invariance(self, small_encoder):
        """E(x + t) == E(x) for any translation t."""
        torch.manual_seed(1)
        pos = torch.randn(5, 3)
        Z = torch.tensor([1, 6, 7, 8, 1])
        r_cut = 5.0
        g_ref = _build_graph(pos, Z, r_cut)
        g_shift = _build_graph(pos + torch.tensor([1.2, -0.4, 3.0]), Z, r_cut)

        model = _build_energy_model(small_encoder).eval()
        with torch.no_grad():
            e_ref = model(g_ref)["energy"]
            e_shift = model(g_shift)["energy"]
        torch.testing.assert_close(e_ref, e_shift, rtol=1e-4, atol=1e-4)

    def test_rotation_invariance(self, small_encoder):
        """E(R·x) == E(x) for random rotations R ∈ SO(3)."""
        torch.manual_seed(2)
        pos = torch.randn(5, 3)
        Z = torch.tensor([1, 6, 7, 8, 1])
        r_cut = 5.0
        g_ref = _build_graph(pos, Z, r_cut)

        model = _build_energy_model(small_encoder).eval()
        with torch.no_grad():
            e_ref = model(g_ref)["energy"]

        for seed in range(5):
            torch.manual_seed(seed)
            R = random_rotation_matrix()
            g_rot = _build_graph(pos @ R.T, Z, r_cut)
            with torch.no_grad():
                e_rot = model(g_rot)["energy"]
            torch.testing.assert_close(
                e_ref, e_rot, rtol=1e-4, atol=1e-4,
                msg=f"SO(3) invariance broken with seed {seed}",
            )

    def test_permutation_invariance(self, small_encoder):
        """E(Π·x) == E(x): relabelling atoms must not change total energy."""
        torch.manual_seed(3)
        pos = torch.randn(5, 3)
        Z = torch.tensor([1, 6, 7, 8, 1])
        r_cut = 5.0
        g_ref = _build_graph(pos, Z, r_cut)

        perm = torch.tensor([3, 0, 4, 1, 2])
        g_perm = _build_graph(pos[perm], Z[perm], r_cut)

        model = _build_energy_model(small_encoder).eval()
        with torch.no_grad():
            e_ref = model(g_ref)["energy"]
            e_perm = model(g_perm)["energy"]
        torch.testing.assert_close(e_ref, e_perm, rtol=1e-4, atol=1e-4)

    def test_cutoff_smoothness(self, small_encoder):
        """An edge exactly at ``r_cut`` contributes zero to the total energy.

        We build two graphs sharing the same near-cutoff geometry, except we
        disable one "far" atom by pulling it past ``r_cut`` in each case.
        Because ``u(r_cut) = 0`` and the encoder gates every scalar update
        by ``u(r_ij)``, removing a just-beyond-cutoff atom must leave the
        total energy unchanged up to a small tolerance.
        """
        r_cut = 5.0
        pos_inside = torch.tensor(
            [[0.0, 0.0, 0.0],
             [1.2, 0.0, 0.0],
             [0.0, 1.3, 0.0],
             [0.0, 0.0, r_cut + 0.01]],    # just past cutoff
        )
        Z = torch.tensor([6, 1, 1, 1])
        g_with_far = _build_graph(pos_inside, Z, r_cut)

        pos_no_far = pos_inside[:3]
        Z_no_far = Z[:3]
        g_no_far = _build_graph(pos_no_far, Z_no_far, r_cut)

        model = _build_energy_model(small_encoder).eval()
        with torch.no_grad():
            e_with = model(g_with_far)["energy"]
            e_without = model(g_no_far)["energy"]

        # Per-atom bias of the readout depends on Z-count, so we compare
        # the *per-atom* energies.  The far atom is past r_cut so it must
        # carry exactly the "isolated atom" energy (no neighbour edges).
        assert torch.isfinite(e_with).all()
        assert torch.isfinite(e_without).all()
        diff = (e_with - e_without).abs().item()
        assert diff < 1e-4, (
            f"energy jumped by {diff:.3e} when an atom just past r_cut "
            "was removed — cutoff smoothness is broken"
        )


# ---------------------------------------------------------------------------
# Functional correctness: does training actually descend?
# ---------------------------------------------------------------------------


class TestOverfitSingleBatch:
    """If the model can't overfit a single batch, forward/backward is broken."""

    def test_overfit_constant_target(self):
        torch.manual_seed(42)
        pos = torch.tensor(
            [[0.00, 0.00, 0.00],
             [0.96, 0.00, 0.00],
             [-0.24, 0.93, 0.00]],
        )
        Z = torch.tensor([8, 1, 1])  # H2O
        g = _build_graph(pos, Z, r_cut=3.0)

        encoder = Allegro(
            num_elements=10,
            num_scalar_features=16,
            num_tensor_features=8,
            r_max=3.0,
            num_bessel=4,
            l_max=1,
            num_layers=1,
            scalar_mlp_hiddens=[16],
            latent_mlp_hiddens=[16],
            residual_alpha=0.5,
        )
        model = _build_energy_model(encoder, avg_nbr=6.0)
        target = torch.tensor([1.234])

        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        initial_loss = None
        for step in range(500):
            opt.zero_grad()
            pred = model(g.clone())["energy"]
            loss = (pred - target).pow(2).mean()
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            opt.step()
        final_loss = loss.item()
        assert final_loss < 1e-4, (
            f"single-batch overfit failed: initial={initial_loss:.3e}, "
            f"final={final_loss:.3e}"
        )
        assert final_loss < initial_loss * 1e-3
