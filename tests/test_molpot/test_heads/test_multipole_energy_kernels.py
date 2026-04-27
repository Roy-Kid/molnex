"""Numerical-validation tests for the ``qm`` and ``mm`` permanent-multipole
energy kernels in :class:`PermMultipoleHead`.

For each kernel we hand-derive the unordered-pair energy from Stone's
textbook formula (Stone, *The Theory of Intermolecular Forces*, 2nd ed.,
2013, §3.3 / Eq. 3.3.5) on a 2-atom system with known geometry, then call
the private kernel directly with synthetic ``q`` / ``μ`` and check the
result matches to float-32 tolerance.

Why test the private kernels directly: the public ``forward`` derives
``q`` / ``μ`` from the encoder's MLP heads, whose weights are random and
not amenable to closed-form reasoning. The kernel itself is the physics
that needs validation; the readouts are equivariance-tested elsewhere.

In addition: a translation + rotation invariance test on the **composed**
Allegro → ``PermMultipoleHead`` pipeline with ``energy_terms=("qq", "qm",
"mm")`` enabled so we know all three kernels remain SO(3)-invariant when
the moments come from the encoder rather than from synthetic input.
"""

from __future__ import annotations

import math

import cuequivariance as cue
import pytest
import torch

from molix.data.types import GraphBatch
from molpot.heads import PermMultipoleHead
from molrep.utils.equivariance import random_rotation_matrix
from molzoo import Allegro
from tests.symmetry_helpers import (
    make_graph_batch,
    rotate_graph,
    translate_graph,
)

SEEDS = (0, 1, 2, 3, 4)
COULOMB = 14.399645  # eV·Å·e⁻²


# ---------------------------------------------------------------------------
# Minimal head builder for unit-testing the private kernels
# ---------------------------------------------------------------------------


def _kernel_test_head(
    *,
    energy_terms: tuple[str, ...],
    charge: bool = True,
    dipole: bool = True,
    quadrupole: bool = False,
) -> PermMultipoleHead:
    """Return a head whose only job is to expose ``_coulomb_qq/qm/mm``.

    The MLP weights inside the head are unused — kernel tests pass
    ``q`` / ``μ`` straight to the private methods.  Constructed with
    ``constrain_total_charge=False`` so the test doesn't need to inject
    a per-graph ``total_charge`` key.
    """
    tensor_irreps = cue.Irreps(cue.O3, [(2, "1o")]) if dipole or quadrupole else None
    return PermMultipoleHead(
        input_dim=4,
        avg_num_neighbors=4.0,
        charge=charge,
        dipole=dipole,
        quadrupole=quadrupole,
        energy_terms=energy_terms,
        constrain_total_charge=False,
        hidden_dim=8,
        cutoff=None,
        damping="none",
        coulomb_constant=COULOMB,
        tensor_irreps=tensor_irreps,
    )


def _two_atom_edges(d: float) -> dict[str, torch.Tensor]:
    """Bidirectional 2-atom edge data with separation ``d`` along x.

    ``edge_index[:, 0]`` is the source, ``edge_index[:, 1]`` the
    destination, matching the molnex edge convention
    (``bond_diff = pos[dst] − pos[src]``).
    """
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    bond_diff = torch.tensor(
        [[+d, 0.0, 0.0], [-d, 0.0, 0.0]],
        dtype=torch.float64,
    )
    bond_dist = torch.tensor([d, d], dtype=torch.float64)
    atom_batch = torch.tensor([0, 0], dtype=torch.long)
    return dict(
        edge_index=edge_index,
        bond_diff=bond_diff,
        bond_dist=bond_dist,
        atom_batch=atom_batch,
    )


# ---------------------------------------------------------------------------
# Analytical 2-atom kernel checks
# ---------------------------------------------------------------------------


class TestQMKernelAnalytical:
    """qm: ``Σ_{i<j} [q_j (R̂·μ_i) − q_i (R̂·μ_j)] / r²`` (Stone §3.3)."""

    def test_aligned_charge_dipole(self):
        """μ_a along R̂, q_b on the other side ⇒ sign matches Stone."""
        head = _kernel_test_head(energy_terms=("qm",))
        e = _two_atom_edges(d=2.0)

        q = torch.tensor([0.5, -0.5], dtype=torch.float64)
        mu = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        )

        # Stone form for unordered pair {0, 1} with R̂_{01} = (1,0,0):
        #   U_qm = (q_1·(R̂·μ_0) − q_0·(R̂·μ_1)) / r²
        #        = (−0.5·1 − 0.5·0) / 4 = −0.125
        expected = COULOMB * -0.125

        out = head._coulomb_qm(
            q.float(),
            mu.float(),
            e["edge_index"],
            e["bond_diff"].float(),
            e["bond_dist"].float(),
            e["atom_batch"],
            n_graphs=1,
        )
        assert out.shape == (1,)
        assert math.isclose(out.item(), expected, rel_tol=1e-5, abs_tol=1e-5)

    def test_zero_when_dipoles_orthogonal_to_axis(self):
        """μ_a, μ_b both perpendicular to R̂ AND parallel to each other.

        Then (R̂·μ_a) = (R̂·μ_b) = 0 ⇒ U_qm = 0.
        """
        head = _kernel_test_head(energy_terms=("qm",))
        e = _two_atom_edges(d=3.0)

        q = torch.tensor([0.7, -0.4], dtype=torch.float64)
        mu = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        out = head._coulomb_qm(
            q.float(),
            mu.float(),
            e["edge_index"],
            e["bond_diff"].float(),
            e["bond_dist"].float(),
            e["atom_batch"],
            n_graphs=1,
        )
        assert abs(out.item()) < 1e-5

    def test_kernel_antisymmetric_under_label_swap(self):
        """qm flips sign under (q,μ)-label swap with positions held fixed.

        Stone qm has the form ``q_b(R̂·μ_a) − q_a(R̂·μ_b)`` — the two
        terms exchange when we relabel ``a ↔ b`` but keep ``R̂``
        pointing the same way, so the kernel value flips sign.  This is
        a structural check on the formula (positions are NOT swapped;
        the relabeled system is physically *different*).
        """
        head = _kernel_test_head(energy_terms=("qm",))
        e = _two_atom_edges(d=2.5)

        q1 = torch.tensor([0.6, -0.3], dtype=torch.float64)
        mu1 = torch.tensor(
            [[0.4, 0.7, -0.2], [-0.5, 0.1, 0.8]],
            dtype=torch.float64,
        )

        a = head._coulomb_qm(
            q1.float(),
            mu1.float(),
            e["edge_index"],
            e["bond_diff"].float(),
            e["bond_dist"].float(),
            e["atom_batch"],
            n_graphs=1,
        ).item()
        b = head._coulomb_qm(
            q1.flip(0).float(),
            mu1.flip(0).float(),
            e["edge_index"],
            e["bond_diff"].float(),
            e["bond_dist"].float(),
            e["atom_batch"],
            n_graphs=1,
        ).item()
        assert math.isclose(a, -b, rel_tol=1e-5, abs_tol=1e-5)

    def test_physical_atom_swap_invariance(self):
        """Physically swapping atoms 0↔1 (positions + (q,μ) together)
        leaves the energy invariant — that's the actual permutation
        symmetry of an unordered pair.

        Implemented by flipping ``bond_diff`` (so R̂ now points 1→0
        instead of 0→1) AND swapping the (q, μ) arrays.  Both flips
        cancel: the kernel sees the same physical configuration.
        """
        head = _kernel_test_head(energy_terms=("qm",))
        e_orig = _two_atom_edges(d=2.5)
        e_swap = dict(e_orig)
        e_swap["bond_diff"] = -e_orig["bond_diff"]

        q1 = torch.tensor([0.6, -0.3], dtype=torch.float64)
        mu1 = torch.tensor(
            [[0.4, 0.7, -0.2], [-0.5, 0.1, 0.8]],
            dtype=torch.float64,
        )

        a = head._coulomb_qm(
            q1.float(),
            mu1.float(),
            e_orig["edge_index"],
            e_orig["bond_diff"].float(),
            e_orig["bond_dist"].float(),
            e_orig["atom_batch"],
            n_graphs=1,
        ).item()
        b = head._coulomb_qm(
            q1.flip(0).float(),
            mu1.flip(0).float(),
            e_swap["edge_index"],
            e_swap["bond_diff"].float(),
            e_swap["bond_dist"].float(),
            e_swap["atom_batch"],
            n_graphs=1,
        ).item()
        assert math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)


class TestMMKernelAnalytical:
    """mm: ``Σ_{i<j} [μ_i·μ_j − 3 (μ_i·R̂)(μ_j·R̂)] / r³`` (Stone §3.3)."""

    def test_parallel_along_axis(self):
        """Both μ's along R̂, magnitude 1 each ⇒ U_mm = (1 − 3) / r³ = −2/r³."""
        head = _kernel_test_head(energy_terms=("mm",), charge=False)
        e = _two_atom_edges(d=2.0)

        mu = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        # U_mm = (1·1 − 3·(1·1)·(1·1)) / 8 = −2/8 = −0.25
        expected = COULOMB * -0.25

        out = head._coulomb_mm(
            mu.float(),
            e["edge_index"],
            e["bond_diff"].float(),
            e["bond_dist"].float(),
            e["atom_batch"],
            n_graphs=1,
        )
        assert math.isclose(out.item(), expected, rel_tol=1e-5, abs_tol=1e-5)

    def test_antiparallel_along_axis(self):
        """μ_a along R̂, μ_b antiparallel ⇒ U_mm = (−1 + 3) / r³ = +2/r³."""
        head = _kernel_test_head(energy_terms=("mm",), charge=False)
        e = _two_atom_edges(d=2.0)

        mu = torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        expected = COULOMB * +0.25

        out = head._coulomb_mm(
            mu.float(),
            e["edge_index"],
            e["bond_diff"].float(),
            e["bond_dist"].float(),
            e["atom_batch"],
            n_graphs=1,
        )
        assert math.isclose(out.item(), expected, rel_tol=1e-5, abs_tol=1e-5)

    def test_perpendicular_to_axis(self):
        """μ_a, μ_b both ⊥ R̂ and parallel ⇒ U_mm = (1 − 0) / r³ = +1/r³."""
        head = _kernel_test_head(energy_terms=("mm",), charge=False)
        e = _two_atom_edges(d=2.0)

        mu = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        # U_mm = (1·1 − 3·0·0) / 8 = 1/8 = 0.125
        expected = COULOMB * 0.125

        out = head._coulomb_mm(
            mu.float(),
            e["edge_index"],
            e["bond_diff"].float(),
            e["bond_dist"].float(),
            e["atom_batch"],
            n_graphs=1,
        )
        assert math.isclose(out.item(), expected, rel_tol=1e-5, abs_tol=1e-5)

    def test_swap_atom_labels(self):
        """mm is symmetric under {a,b} swap (no sign flip — both R̂ factors
        flip together)."""
        head = _kernel_test_head(energy_terms=("mm",), charge=False)
        e = _two_atom_edges(d=3.5)

        mu1 = torch.tensor(
            [[0.4, 0.7, -0.2], [-0.5, 0.1, 0.8]],
            dtype=torch.float64,
        )
        mu2 = mu1.flip(0)

        a = head._coulomb_mm(
            mu1.float(),
            e["edge_index"],
            e["bond_diff"].float(),
            e["bond_dist"].float(),
            e["atom_batch"],
            n_graphs=1,
        ).item()
        b = head._coulomb_mm(
            mu2.float(),
            e["edge_index"],
            e["bond_diff"].float(),
            e["bond_dist"].float(),
            e["atom_batch"],
            n_graphs=1,
        ).item()
        assert math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


class TestEnergyTermValidation:
    """Term × moment compatibility is enforced at construction."""

    def test_qm_requires_dipole(self):
        with pytest.raises(ValueError, match="dipole"):
            PermMultipoleHead(
                input_dim=4,
                avg_num_neighbors=4.0,
                charge=True,
                dipole=False,
                energy_terms=("qq", "qm"),
                constrain_total_charge=False,
                hidden_dim=8,
            )

    def test_mm_requires_dipole(self):
        with pytest.raises(ValueError, match="dipole"):
            PermMultipoleHead(
                input_dim=4,
                avg_num_neighbors=4.0,
                charge=True,
                dipole=False,
                energy_terms=("mm",),
                constrain_total_charge=False,
                hidden_dim=8,
            )

    def test_unimplemented_qt_mt_tt_raise(self):
        for term in ("qt", "mt", "tt"):
            with pytest.raises(NotImplementedError, match=term):
                PermMultipoleHead(
                    input_dim=4,
                    avg_num_neighbors=4.0,
                    charge=True,
                    dipole=True,
                    quadrupole=True,
                    energy_terms=("qq", term),
                    constrain_total_charge=False,
                    hidden_dim=8,
                    tensor_irreps=cue.Irreps(cue.O3, [(2, "1o"), (2, "2e")]),
                )

    def test_old_names_rejected(self):
        """Pre-rename keys (``qmu`` / ``mumu`` / ...) must be unknown."""
        for old in ("qmu", "mumu", "qtheta", "mutheta", "thetatheta"):
            with pytest.raises(ValueError, match="Unknown energy_terms"):
                PermMultipoleHead(
                    input_dim=4,
                    avg_num_neighbors=4.0,
                    charge=True,
                    dipole=True,
                    energy_terms=("qq", old),
                    constrain_total_charge=False,
                    hidden_dim=8,
                    tensor_irreps=cue.Irreps(cue.O3, [(2, "1o")]),
                )


# ---------------------------------------------------------------------------
# Composed pipeline: translation + rotation invariance with all 3 terms
# ---------------------------------------------------------------------------


@pytest.fixture
def small_molecule_neutral() -> GraphBatch:
    """Same fixture shape as test_multipole_symmetry — 5-atom, 2 graphs."""
    torch.manual_seed(42)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.3, 0.0],
            [2.5, 0.0, 0.1],
            [4.0, 0.5, 0.0],
            [5.3, 0.2, 0.1],
        ],
        dtype=torch.float32,
    )
    Z = torch.tensor([6, 1, 8, 6, 1], dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 1],
            [3, 4],
            [4, 3],
        ],
        dtype=torch.long,
    )
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    total_charge = torch.tensor([0.0, 0.0], dtype=torch.float32)
    return make_graph_batch(
        pos=pos,
        Z=Z,
        edge_index=edge_index,
        batch=batch,
        graphs={"total_charge": total_charge},
    )


@pytest.fixture
def charge_dipole_full_terms_pipeline():
    """Full-terms pipeline: q + μ heads on, energy = qq + qm + mm.

    Built locally because the existing fixtures pin ``energy_terms=("qq",)``.
    """
    import torch.nn as nn

    from tests.symmetry_helpers import recompute_edge_geometry

    torch.manual_seed(0)
    encoder = Allegro(
        num_elements=10,
        num_scalar_features=16,
        num_tensor_features=8,
        r_max=8.0,
        num_layers=2,
        type_embed_dim=16,
        latent_mlp_depth=1,
        latent_mlp_width=16,
        avg_num_neighbors=4.0,
        expose_tensor_track=True,
    )
    encoder.eval()
    head = PermMultipoleHead(
        input_dim=encoder.num_scalar_features * (encoder.num_layers + 1),
        avg_num_neighbors=4.0,
        charge=True,
        dipole=True,
        quadrupole=False,
        energy_terms=("qq", "qm", "mm"),
        constrain_total_charge=True,
        hidden_dim=16,
        tensor_irreps=encoder.tensor_track_irreps,
    ).eval()

    class _Pipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(self, batch):
            recompute_edge_geometry(batch)
            batch = self.encoder(batch)
            return self.head(batch)

    return _Pipeline()


class TestComposedPipelineInvariance:
    """All three implemented terms (qq + qm + mm) must keep the energy
    translation- and rotation-invariant when the moments come from the
    encoder."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_translation(self, charge_dipole_full_terms_pipeline, small_molecule_neutral, seed):
        torch.manual_seed(seed)
        t = torch.randn(3) * 10.0

        with torch.no_grad():
            ref = charge_dipole_full_terms_pipeline(small_molecule_neutral.clone())
            shifted = charge_dipole_full_terms_pipeline(translate_graph(small_molecule_neutral, t))
        assert torch.allclose(
            ref["energy_es"],
            shifted["energy_es"],
            atol=1e-4,
            rtol=1e-4,
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_rotation(self, charge_dipole_full_terms_pipeline, small_molecule_neutral, seed):
        torch.manual_seed(seed)
        R = random_rotation_matrix()

        with torch.no_grad():
            ref = charge_dipole_full_terms_pipeline(small_molecule_neutral.clone())
            rot = charge_dipole_full_terms_pipeline(rotate_graph(small_molecule_neutral, R))
        assert torch.allclose(
            ref["energy_es"],
            rot["energy_es"],
            atol=1e-4,
            rtol=1e-4,
        )
