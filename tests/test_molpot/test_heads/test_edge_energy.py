"""Tests for ``molpot.heads.EdgeEnergyHead``."""

from __future__ import annotations

import math

import pytest
import torch

from molix.data.types import AtomData, EdgeData, GraphBatch, GraphData
from molpot.heads import EdgeEnergyHead


def _stub_batch(*, n_atoms=3, n_edges=6, n_layers=2, feat_dim=8):
    torch.manual_seed(0)
    edge_index = torch.tensor(
        [[0, 1], [1, 0], [1, 2], [2, 1], [0, 2], [2, 0]], dtype=torch.long
    )[:n_edges]
    atoms = AtomData(
        Z=torch.tensor([1, 6, 8][:n_atoms]),
        pos=torch.randn(n_atoms, 3),
        batch=torch.zeros(n_atoms, dtype=torch.long),
        batch_size=[n_atoms],
    )
    edges = EdgeData(
        edge_index=edge_index,
        bond_diff=torch.randn(n_edges, 3),
        bond_dist=torch.rand(n_edges) + 0.5,
        batch_size=[n_edges],
    )
    edges["edge_features"] = torch.randn(n_edges, n_layers, feat_dim)
    graphs = GraphData(num_atoms=torch.tensor([n_atoms]), batch_size=[1])
    return GraphBatch(atoms=atoms, edges=edges, graphs=graphs, batch_size=[])


class TestShape:
    def test_returns_per_graph_energy(self):
        head = EdgeEnergyHead(
            input_dim=8, hidden_dim=4, avg_num_neighbors=3.0
        )
        out = head(_stub_batch())
        assert set(out.keys()) == {"energy"}
        assert out["energy"].shape == (1,)

    def test_custom_out_key(self):
        head = EdgeEnergyHead(
            input_dim=8, hidden_dim=4, avg_num_neighbors=3.0, out_key="E"
        )
        out = head(_stub_batch())
        assert "E" in out


class TestLayerReduction:
    def test_last_picks_final_layer(self):
        head = EdgeEnergyHead(
            input_dim=8, hidden_dim=4, avg_num_neighbors=3.0, layer_reduction="last"
        )
        b = _stub_batch(n_layers=3)
        # Manually reduce to what "last" should see.
        feats = b["edges", "edge_features"][:, -1]
        expected_edge_e = head.mlp(feats).squeeze(-1) / math.sqrt(3.0)
        # We can't easily reproduce the scatter, but we can check that a
        # single-layer batch gives the same result as a 3-layer batch with
        # identical last-layer content.
        b_single = _stub_batch(n_layers=1)
        b_single["edges", "edge_features"] = feats.unsqueeze(1)
        with torch.no_grad():
            e_multi = head(b)["energy"]
            e_single = head(b_single)["energy"]
        torch.testing.assert_close(e_multi, e_single, rtol=1e-5, atol=1e-5)

    def test_sum_equals_sum_of_layer_features(self):
        """``sum`` reduction must see exactly the sum of per-layer feature tensors."""
        head = EdgeEnergyHead(
            input_dim=8, hidden_dim=4, avg_num_neighbors=3.0,
            layer_reduction="sum",
        )
        b = _stub_batch(n_layers=3)
        b_flat = _stub_batch(n_layers=1)
        b_flat["edges", "edge_features"] = (
            b["edges", "edge_features"].sum(dim=1).unsqueeze(1)
        )
        with torch.no_grad():
            e_multi = head(b)["energy"]
            e_flat = head(b_flat)["energy"]
        torch.testing.assert_close(e_multi, e_flat, rtol=1e-5, atol=1e-5)

    def test_reject_invalid_reduction(self):
        with pytest.raises(ValueError, match="layer_reduction"):
            EdgeEnergyHead(
                input_dim=8,
                hidden_dim=4,
                avg_num_neighbors=3.0,
                layer_reduction="max",  # not supported
            )


class TestNeighborNormalization:
    def test_sqrt_normalization_is_applied(self):
        """Halving ``avg_num_neighbors`` scales edge energy by √2."""
        b = _stub_batch()
        # two heads with different avg_num_neighbors but identical parameters
        h1 = EdgeEnergyHead(input_dim=8, hidden_dim=4, avg_num_neighbors=4.0)
        h2 = EdgeEnergyHead(input_dim=8, hidden_dim=4, avg_num_neighbors=16.0)
        h2.load_state_dict(h1.state_dict())
        with torch.no_grad():
            e1 = h1(b)["energy"]
            e2 = h2(b)["energy"]
        # √16 / √4 = 2 → e1 should be twice e2.
        torch.testing.assert_close(e1, 2.0 * e2, rtol=1e-5, atol=1e-5)

    def test_per_source_fallback_runs(self):
        head = EdgeEnergyHead(input_dim=8, hidden_dim=4, avg_num_neighbors=None)
        out = head(_stub_batch())
        assert out["energy"].shape == (1,)
        assert torch.isfinite(out["energy"]).all()

    def test_rejects_non_positive_avg_num_neighbors(self):
        with pytest.raises(ValueError, match="avg_num_neighbors"):
            EdgeEnergyHead(input_dim=8, hidden_dim=4, avg_num_neighbors=0.0)


class TestEmptyGraph:
    def test_zero_edges_give_zero_energy(self):
        """Graph with no edges → scatter sum is zero tensor."""
        atoms = AtomData(
            Z=torch.tensor([1]),
            pos=torch.zeros(1, 3),
            batch=torch.zeros(1, dtype=torch.long),
            batch_size=[1],
        )
        edges = EdgeData(
            edge_index=torch.empty(0, 2, dtype=torch.long),
            bond_diff=torch.empty(0, 3),
            bond_dist=torch.empty(0),
            batch_size=[0],
        )
        edges["edge_features"] = torch.empty(0, 2, 8)
        graphs = GraphData(num_atoms=torch.tensor([1]), batch_size=[1])
        b = GraphBatch(atoms=atoms, edges=edges, graphs=graphs, batch_size=[])

        head = EdgeEnergyHead(input_dim=8, hidden_dim=4, avg_num_neighbors=1.0)
        with torch.no_grad():
            out = head(b)
        assert out["energy"].shape == (1,)
        assert out["energy"].item() == 0.0
