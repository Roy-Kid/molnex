"""Sonata — static permanent-electrostatics potential model.

Sonata composes an Allegro-style encoder with a perm-multipole head and
σ-screened Ewald multipole-Coulomb evaluator, plus an optional short-range
energy head. It is the molzoo model for the Cheng 2025 LES architecture.

References:
    * Cheng B., *Latent Ewald summation for machine-learning potentials*,
      npj Comput. Mater. **11**, 80 (2025).
      https://doi.org/10.1038/s41524-025-01577-7
    * Aguado A. & Madden P. A., *Ewald summation of electrostatic
      multipole interactions up to the quadrupolar level*,
      J. Chem. Phys. **119**, 7471 (2003).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict

from molix.data.types import GraphBatch
from molpot.heads.multipole import PermMultipoleHead, PermMultipoleHeadSpec
from molpot.potentials.electrostatics.ewald import (
    EwaldMultipoleEnergy,
    EwaldMultipoleEnergySpec,
)
from molpot.potentials.polarization import (
    Polarization,
    _set_les_electrostatics_precomputed,
)

# ---------------------------------------------------------------------------
# Refusal messages
# ---------------------------------------------------------------------------

_INDUCED_REFUSE_MSG = (
    "Sonata is the *static permanent-electrostatics* composer; "
    "induced-response inputs ({name}={value!r}) belong to the future "
    "`LesPolarizable` composer line, which wraps `EwaldMultipoleEnergy` "
    "with κ / α heads for inline non-self-consistent linear response. "
    "Use that composer (when implemented) for induced-response training."
)

_POLARIZATION_REFUSE_MSG = (
    "Sonata refuses `Polarization` in `short_range_head`: it would "
    "double-count induction against any future inline LES α-mode. "
    "Polarization belongs to the `LesPolarizable` composer line; use "
    "an `EdgeEnergyHead` (or any head that writes a dict containing "
    "`'energy_short'`) here instead."
)


def _theta_to_cartesian_quadrupole(theta: torch.Tensor) -> torch.Tensor:
    """Convert ``(N, 5)`` ℓ=2 real-spherical components to ``(N, 3, 3)``.

    The result is a symmetric, traceless 3×3 Cartesian tensor that is
    rotation-equivariant with the input under cuequivariance's D⁽²⁾
    Wigner matrix.

    The cuequivariance ``2e`` real-spherical-harmonic convention maps the
    five ir_mul components to:

    * index 0 (m=-2): ``√15 · xz``
    * index 1 (m=-1): ``√15 · xy``
    * index 2 (m=0):  ``(√5/2) · (-x² + 2y² - z²)``
    * index 3 (m=+1): ``√15 · yz``
    * index 4 (m=+2): ``(√15/2) · (z² - x²)``
    """
    s2 = 0.5**0.5
    s6 = 6.0**-0.5
    n = theta.shape[0]
    q = torch.zeros(n, 3, 3, dtype=theta.dtype, device=theta.device)
    q[:, 0, 2] = q[:, 2, 0] = theta[:, 0] * s2
    q[:, 0, 1] = q[:, 1, 0] = theta[:, 1] * s2
    q[:, 0, 0] = -theta[:, 2] * s6
    q[:, 1, 1] = 2.0 * theta[:, 2] * s6
    q[:, 2, 2] = -theta[:, 2] * s6
    q[:, 1, 2] = q[:, 2, 1] = theta[:, 3] * s2
    q[:, 0, 0] = q[:, 0, 0] - theta[:, 4] * s2
    q[:, 2, 2] = q[:, 2, 2] + theta[:, 4] * s2
    return q


# ---------------------------------------------------------------------------
# SonataSpec
# ---------------------------------------------------------------------------


class SonataSpec(BaseModel):
    """Frozen Pydantic snapshot of a :class:`Sonata` model.

    Attributes:
        head: :class:`PermMultipoleHeadSpec` for the multipole readout.
        ewald: :class:`EwaldMultipoleEnergySpec` for the Ewald evaluator.
        short_range: ``None`` when no short-range head is wired, else an
            opaque metadata dict.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    head: PermMultipoleHeadSpec
    ewald: EwaldMultipoleEnergySpec
    short_range: dict | None = None


# ---------------------------------------------------------------------------
# Sonata
# ---------------------------------------------------------------------------


class Sonata(nn.Module):
    """Static permanent-electrostatics potential with Ewald long-range.

    Wires an Allegro-style encoder → :class:`PermMultipoleHead` →
    :class:`EwaldMultipoleEnergy` → optional short-range head.

    Args:
        encoder: Allegro-style encoder. Must be constructed with
            ``expose_tensor_track=True``; ``l_max ≥ 2`` is required when
            ``dipole=True`` or ``quadrupole=True``.
        sigma: σ-Gaussian charge-smearing length. Unit: Å. Default 1.0.
        dl: Reciprocal-space grid resolution. Unit: Å. Default 2.0.
        prefactor: Electrostatic prefactor ``1/(2 ε₀)``.
            Unit: eV·Å·e⁻². Default 90.4756.
        charge: Predict atomic charges. Default True.
        dipole: Predict atomic dipoles. Requires ``encoder.l_max >= 2``.
        quadrupole: Predict atomic quadrupoles. Requires ``encoder.l_max >= 2``.
        constrain_total_charge: Project per-graph charge sums onto the
            ``total_charge_key`` field. Default True.
        short_range_head: Optional module writing ``"energy_short"`` ``(B,)``.
        total_charge_key: Per-graph total-charge key under ``graphs``.
        hidden_dim: Hidden width of the multipole head's scalar MLPs.
        avg_num_neighbors: Dataset-wide ⟨|N(i)|⟩ for the edge→atom pool
            normalisation in the multipole head.
        compute_forces: Derive forces ``F = -∂U/∂pos`` via autograd in
            :meth:`forward`. Default ``False``.
        compute_stress: Derive stress ``σ = (1/V) ∂U/∂ε`` via strain
            perturbation in :meth:`forward`. Requires ``("graphs", "cell")``.
            Default ``False``.

    Forward output (flat dict):
        ``energy`` ``(B,)`` — total per-graph energy.
        ``energy_short`` ``(B,)`` — short-range contribution (zeros when no SR head).
        ``energy_es`` ``(B,)`` — electrostatic (Ewald) energy.
        ``atomic_charges`` ``(N,)`` — per-atom charges. Unit: e.
        ``atomic_dipoles`` ``(N, 3)`` — per-atom dipoles (when dipole=True). Unit: e·Å.
        ``atomic_quadrupoles`` ``(N, 5)`` — per-atom quadrupoles in traceless-symmetric
        spherical basis (when quadrupole=True). Unit: e·Å².
        ``molecular_dipole`` ``(B, 3)`` — molecular dipole moment. Unit: e·Å.
        ``phi`` ``(N,)`` — per-atom electrostatic potential. Unit: V.
        ``field`` ``(N, 3)`` — per-atom electric field. Unit: V·Å⁻¹.
        ``charge_sum_pre_proj`` / ``charge_sum_post_proj`` — per-graph charge sums
        before/after total-charge projection.
        ``forces`` ``(N, 3)`` — only when ``self.compute_forces=True``. Unit: eV·Å⁻¹.
        ``stress`` ``(B, 3, 3)`` — only when ``self.compute_stress=True``. Unit: eV·Å⁻³.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        sigma: float = 1.0,
        dl: float = 2.0,
        prefactor: float = 90.4756,
        charge: bool = True,
        dipole: bool = True,
        quadrupole: bool = True,
        constrain_total_charge: bool = True,
        short_range_head: nn.Module | list[nn.Module] | None = None,
        total_charge_key: str = "total_charge",
        hidden_dim: int = 128,
        avg_num_neighbors: float | None = None,
        compute_forces: bool = False,
        compute_stress: bool = False,
        **kwargs: Any,
    ) -> None:
        # --- scope-refusal checks (before super().__init__) ---
        if "kappa_head" in kwargs:
            raise ValueError(
                _INDUCED_REFUSE_MSG.format(name="kappa_head", value=kwargs["kappa_head"])
            )
        if "alpha_head" in kwargs:
            raise ValueError(
                _INDUCED_REFUSE_MSG.format(name="alpha_head", value=kwargs["alpha_head"])
            )
        for name in kwargs:
            if name.startswith("induced_"):
                raise ValueError(_INDUCED_REFUSE_MSG.format(name=name, value=kwargs[name]))
        if kwargs:
            raise TypeError(f"Sonata.__init__ got unexpected kwargs: {sorted(kwargs)}")

        if isinstance(short_range_head, Polarization):
            raise ValueError(_POLARIZATION_REFUSE_MSG)
        if isinstance(short_range_head, (list, nn.ModuleList)) and any(
            isinstance(h, Polarization) for h in short_range_head
        ):
            raise ValueError(_POLARIZATION_REFUSE_MSG)

        # --- encoder pre-checks ---
        if not getattr(encoder, "expose_tensor_track", False):
            raise ValueError(
                "Sonata requires an encoder built with `expose_tensor_track=True` "
                "to expose the equivariant tensor-track features the multipole "
                "head consumes. Reconstruct your encoder with this flag set."
            )
        if (dipole or quadrupole) and getattr(encoder, "l_max", 0) < 2:
            raise ValueError(
                "Sonata requires `encoder.l_max >= 2` when `dipole=True` or "
                f"`quadrupole=True`; got `l_max={getattr(encoder, 'l_max', None)}`."
            )

        super().__init__()

        self.compute_forces = compute_forces
        self.compute_stress = compute_stress

        # --- sub-modules ---
        self.encoder = encoder
        self.perm_multipole_head = PermMultipoleHead(
            input_dim=encoder.output_dim,
            avg_num_neighbors=avg_num_neighbors,
            charge=charge,
            dipole=dipole,
            quadrupole=quadrupole,
            constrain_total_charge=constrain_total_charge,
            total_charge_key=total_charge_key,
            hidden_dim=hidden_dim,
            tensor_irreps=encoder.tensor_track_irreps,
        )
        self.ewald = EwaldMultipoleEnergy(
            sigma=sigma,
            dl=dl,
            prefactor=prefactor,
            remove_self_interaction=True,
            use_epsilon_r_scaling=False,
        )

        # --- short-range head(s) ---
        self.short_range_head: nn.Module | nn.ModuleList | None
        if short_range_head is None:
            self.short_range_head = None
        elif isinstance(short_range_head, nn.ModuleList):
            self.short_range_head = short_range_head
        elif isinstance(short_range_head, list):
            self.short_range_head = nn.ModuleList(short_range_head)
        else:
            self.short_range_head = short_range_head

        # --- short_range metadata ---
        short_range_meta: dict | None = None
        if isinstance(self.short_range_head, nn.ModuleList):
            short_range_meta = {
                "kind": "list",
                "types": [type(h).__name__ for h in self.short_range_head],
            }
        elif isinstance(self.short_range_head, nn.Module):
            short_range_meta = {
                "kind": "single",
                "type": type(self.short_range_head).__name__,
            }

        self.config = SonataSpec(
            head=self.perm_multipole_head.config,
            ewald=self.ewald.config,
            short_range=short_range_meta,
        )

    @classmethod
    def from_spec(cls, spec: SonataSpec, encoder: nn.Module) -> "Sonata":
        """Construct from a frozen :class:`SonataSpec` and an encoder.

        Extracts loose hyperparameters from the sub-specs and passes them
        through the standard constructor so the head and Ewald evaluator
        are built fresh.

        ``spec.short_range`` is opaque metadata only — the user must
        supply the actual short-range head externally if a round-trip
        wiring with the same head is required.
        """
        return cls(
            encoder=encoder,
            sigma=spec.ewald.sigma,
            dl=spec.ewald.dl,
            prefactor=spec.ewald.prefactor,
            charge=spec.head.charge,
            dipole=spec.head.dipole,
            quadrupole=spec.head.quadrupole,
            constrain_total_charge=spec.head.constrain_total_charge,
            total_charge_key=spec.head.total_charge_key,
            hidden_dim=spec.head.hidden_dim,
            avg_num_neighbors=spec.head.avg_num_neighbors,
        )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: GraphBatch,
        *,
        kvec_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run encoder → multipole head → Ewald → optional short-range.

        Args:
            batch: Post-collate :class:`GraphBatch` carrying ``atoms``,
                ``edges``, and ``graphs`` sub-dicts. Periodic systems
                must include ``("graphs", "cell")``.
            kvec_indices: Optional ``(M, 3)`` precomputed integer triplet
                array for the reciprocal-space grid. Production callers
                must leave this ``None``.

        Returns:
            Flat dict with the keys documented in the class docstring.
        """
        atom_batch = batch["atoms", "batch"]
        edge_index = batch["edges", "edge_index"]

        pos = batch["atoms", "pos"]
        cell = batch.get(("graphs", "cell"))

        cell_orig: torch.Tensor | None = None
        strain: torch.Tensor | None = None
        if self.compute_stress:
            if cell is None:
                raise ValueError("compute_stress=True requires `('graphs', 'cell')` in the batch.")
            cell_orig = cell
            pos_orig = pos.detach()
            if self.compute_forces:
                pos_orig = pos_orig.requires_grad_(True)
            strain = torch.zeros_like(cell_orig, requires_grad=True)
            sym_eps = 0.5 * (strain + strain.transpose(-1, -2))
            eye_b = torch.eye(3, dtype=cell_orig.dtype, device=cell_orig.device).expand_as(
                cell_orig
            )
            i_plus = eye_b + sym_eps
            pos = torch.einsum("aij,aj->ai", i_plus[atom_batch], pos_orig)
            cell = torch.einsum("bij,blj->bli", i_plus, cell_orig)
            pos_for_force_grad: torch.Tensor | None = pos_orig if self.compute_forces else None
        elif self.compute_forces:
            pos = pos.detach().requires_grad_(True)
            pos_for_force_grad = pos
        else:
            pos_for_force_grad = None

        if self.compute_forces or self.compute_stress:
            bond_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
            bond_dist = bond_diff.norm(dim=-1)
            batch[("atoms", "pos")] = pos
            batch[("edges", "bond_diff")] = bond_diff
            batch[("edges", "bond_dist")] = bond_dist
            if self.compute_stress:
                batch[("graphs", "cell")] = cell

        self.encoder(batch)

        head_out = self.perm_multipole_head(batch)

        head = self.perm_multipole_head
        q = batch["atoms", head.out_charge_key]
        mu = batch.get(("atoms", head.out_dipole_key)) if head.dipole else None
        Theta = batch.get(("atoms", head.out_quadrupole_key)) if head.quadrupole else None

        Q_cart = _theta_to_cartesian_quadrupole(Theta) if Theta is not None else None

        ewald_out = self.ewald(
            q=q,
            pos=pos,
            cell=cell,
            batch=atom_batch,
            mu=mu,
            Q=Q_cart,
            kvec_indices=kvec_indices,
        )
        energy_es = ewald_out["pot"]
        if energy_es.dim() == 0:
            energy_es = energy_es.unsqueeze(0)

        n_graphs = int(energy_es.shape[0])

        if self.short_range_head is None:
            energy_short = torch.zeros(n_graphs, dtype=energy_es.dtype, device=energy_es.device)
        else:
            heads_iter = (
                self.short_range_head
                if isinstance(self.short_range_head, nn.ModuleList)
                else [self.short_range_head]
            )
            energy_short = torch.zeros(n_graphs, dtype=energy_es.dtype, device=energy_es.device)
            for sr_head in heads_iter:
                sr_out = sr_head(batch)
                if not isinstance(sr_out, dict):
                    raise TypeError(
                        f"Short-range head {type(sr_head).__name__} must return "
                        f"a dict; got {type(sr_out).__name__}."
                    )
                if "energy_short" not in sr_out:
                    raise KeyError(
                        f"Short-range head {type(sr_head).__name__} must return "
                        f"a dict with key 'energy_short'; got keys {list(sr_out)}."
                    )
                energy_short = energy_short + sr_out["energy_short"]

        energy = energy_short + energy_es

        out: dict[str, torch.Tensor] = {
            "energy": energy,
            "energy_short": energy_short,
            "energy_es": energy_es,
            "atomic_charges": q,
            "phi": ewald_out["phi"],
            "field": ewald_out["field"],
        }
        if mu is not None:
            out["atomic_dipoles"] = mu
        if Theta is not None:
            out["atomic_quadrupoles"] = Theta
        if "molecular_dipole" in head_out:
            out["molecular_dipole"] = head_out["molecular_dipole"]
        if "charge_sum_pre_proj" in head_out:
            out["charge_sum_pre_proj"] = head_out["charge_sum_pre_proj"]
        if "charge_sum_post_proj" in head_out:
            out["charge_sum_post_proj"] = head_out["charge_sum_post_proj"]

        if self.compute_forces:
            assert pos_for_force_grad is not None
            forces = -torch.autograd.grad(
                energy.sum(),
                pos_for_force_grad,
                create_graph=self.training,
                retain_graph=self.compute_stress or self.training,
            )[0]
            out["forces"] = forces

        if self.compute_stress:
            assert strain is not None and cell_orig is not None
            stress_eps = torch.autograd.grad(
                energy.sum(),
                strain,
                create_graph=self.training,
            )[0]
            stress_eps_sym = (
                stress_eps
                + stress_eps.transpose(-1, -2)
                - torch.diag_embed(torch.diagonal(stress_eps, dim1=-2, dim2=-1))
            )
            volume = torch.linalg.det(cell_orig).abs()
            out["stress"] = stress_eps_sym / volume.unsqueeze(-1).unsqueeze(-1)

        _set_les_electrostatics_precomputed(True)

        return out
