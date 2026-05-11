"""Latent Ewald Summation (LES) screened-Coulomb multipole electrostatics.

Single :class:`EwaldMultipoleEnergy` potential implementing the LES paper's
σ-Gaussian-screened Coulomb energy with permanent atomic multipoles
``(q, μ, Q)`` and inline non-self-consistent linear response from
``(κ, α)``.

Algorithm (per Cheng 2025; verified against `github.com/ChengUCB/les`):

* **Periodic systems** (cell present, ``det(cell) > 0``): half-k-sphere
  reciprocal-space sum of the structure factor

  .. math::

      S(\\mathbf{k}) = \\sum_i \\bigl[q_i + i \\mathbf{k}\\!\\cdot\\!\\mu_i
                       - \\tfrac{1}{2} \\mathbf{k}\\!\\cdot\\!Q_i\\!\\cdot\\!\\mathbf{k}\\bigr]
                       e^{i \\mathbf{k}\\cdot\\mathbf{r}_i}

  weighted by ``kfac(k) = exp(-σ²k²/2) / k²`` and an overall prefactor
  ``norm_factor / V``. ``k = 0`` is excluded → uniform compensating
  background (Yeh-Berkowitz tin-foil convention; charge non-neutrality
  is permitted but the boundary is documented).

* **Non-periodic systems** (cell absent or ``det(cell) ≤ 0``): O(N²)
  pairwise sum over the full pre-screened kernels

  .. math::

      f_{qq}(r) = \\frac{\\operatorname{erf}(r \\cdot a)}{r} \\cdot c, \\quad
      a = \\frac{1}{\\sigma \\sqrt{2}}, \\quad c = \\frac{1}{4 \\pi \\varepsilon_0}

  plus higher-order multipole gradients ``f_qu, f_uu, f_Qu, f_QQ``
  derived by successive differentiation of ``f_qq``.

* **Non-self-consistent induced response** (one-shot, no iteration):

  .. math::

      q_{\\text{ind}, i} = -\\kappa_i \\, \\Phi(\\mathbf{r}_i), \\quad
      \\mathbf{u}_{\\text{ind}, i} = \\alpha_i \\, \\mathbf{E}(\\mathbf{r}_i),
      \\quad
      U_{iq} = \\tfrac{1}{2} \\Phi \\cdot q_{\\text{ind}}, \\quad
      U_{iu} = -\\tfrac{1}{2} \\mathbf{E} \\cdot \\mathbf{u}_{\\text{ind}}

  Both Φ and E respect ``remove_self_interaction=True`` so the linear
  response is to the *other-source* field, not the atom's own
  Gaussian. This is **not** the self-consistent CG Thole solve in
  :class:`molpot.potentials.Polarization`; see the "What this is NOT"
  block in :class:`EwaldMultipoleEnergy`.

Self-correction constants (excluded when
``remove_self_interaction=True``):

* charge:      ``q² · √(2/π) / (2σ) · prefactor``
* per-atom Φ:  ``q · 2 / (σ (2π)^(3/2)) · prefactor`` — this is the
  flagged self-term whose exclusion ensures ``Φ_i`` reports only the
  potential due to *other* atoms (the user's hard requirement)
* dipole:      ``‖μ‖² / (3 σ³ (2π)^(3/2)) · prefactor``
* quadrupole:  ``‖Q‖_F² / (10 σ⁵ (2π)^(3/2)) · prefactor``  (assumes Q
  symmetric and traceless — see Flag #2)
* per-atom E:  ``c_self · μ`` with ``c_self = (4/(3√π))·a³ · prefactor / (2π)``

Defaults follow LES upstream: ``σ = 1.0 Å``, ``dl = 2.0 Å``,
``prefactor = 90.4756 eV·Å·e⁻²`` (= ``1/(2 ε₀)`` in eV-Å-e units),
``remove_self_interaction = True``, ``use_epsilon_r_scaling = False``.

What this is NOT:
    * **Not standard Ewald split.** There is no real-space + reciprocal
      partition with the same screening — periodic uses reciprocal
      *only*, non-periodic uses real-space *only*. The
      ``erf(r/(σ√2))/r`` kernel is the **whole** interaction.
    * **Not :class:`molpot.potentials.Polarization`.** That class
      solves a self-consistent CG Thole iteration; LES α-mode is a
      **one-shot** non-self-consistent linear response. Composing both
      in one ``PotentialComposer`` would double-count induction.
    * **Not Qeq / CELLI.** No global Lagrangian solve with KKT or
      Hirshfeld supervision.

Three flagged caveats requiring user awareness:

1. **🟡 Flag #1 — multipole-S(k) provenance.** The multipole structure
   factor and per-atom μ²/Q² self-corrections are NOT present in the
   v1 arXiv preprint of King et al. *Nat. Commun.* 16:8763 (2025)
   (arXiv:2412.15455v1); they appear only in the upstream LES code.
   The canonical published derivation of the multipole-Ewald split
   with explicit per-multipole self-corrections is Aguado & Madden,
   *J. Chem. Phys.* 119, 7471 (2003) — see the references block.
   The implementation here matches `les.module.make_kernels`
   line-by-line; corroboration against the published Nat. Commun. SI
   is pending.

2. **🟢 Flag #2 — Q tracelessness.** The quadrupole self-correction
   ``‖Q‖_F² / (10 σ⁵ (2π)^(3/2))`` assumes Q symmetric and traceless.
   Upstream LES does not enforce this; here we trace-project at the
   readout layer (in :class:`molpot.heads.PolarizabilityHead`'s l=2
   path and in :class:`molpot.heads.PermMultipoleHead`'s Θ output).
   Non-traceless Q passed in directly will silently absorb its trace
   into a renormalisation of σ-monopole.

3. **🟢 Flag #3 — k=0 background.** The k=0 reciprocal-space term is
   excluded. Mathematically this imposes a uniform compensating
   background charge (Yeh-Berkowitz tin-foil convention). Charge
   non-neutrality is permitted but the boundary is documented.

References:

* Cheng B., *Latent Ewald summation for machine-learning potentials*,
  npj Comput. Mater. **11**, 80 (2025).
  https://doi.org/10.1038/s41524-025-01577-7
* King D. S. et al., *Latent equivariant ML force fields with long-range
  electrostatics*, Nat. Commun. **16**, 8763 (2025).
  https://doi.org/10.1038/s41467-025-63852-x
* Aguado A. & Madden P. A., *Ewald summation of electrostatic multipole
  interactions up to the quadrupolar level*, J. Chem. Phys. **119**,
  7471 (2003). https://doi.org/10.1063/1.1605941 — canonical reference
  for the multipole-Ewald split with explicit per-multipole
  self-corrections.
* Stone A. J., *The Theory of Intermolecular Forces*, 2nd ed. (Oxford,
  2013), §3 — multipole interaction kernels.
* Upstream code: https://github.com/ChengUCB/les (treated as oracle for
  the brute-force NumPy parity tests under
  ``tests/_oracles/screened_coulomb.py``).
* Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed. (Oxford,
  2017) §6.5; Frenkel & Smit, *Understanding Molecular Simulation*,
  2nd ed., §12.1; Hünenberger, *Adv. Polym. Sci.* 173:105 (2005)
  §3.2.2; Yeh & Berkowitz, *J. Chem. Phys.* 111:3155 (1999).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field

from molpot.potentials.base import BasePotential

# ---------------------------------------------------------------------------
# Pydantic config
# ---------------------------------------------------------------------------


class EwaldMultipoleEnergySpec(BaseModel):
    """Configuration snapshot for :class:`EwaldMultipoleEnergy`.

    Frozen Pydantic model — every constructor argument is captured here
    so a trained checkpoint carries an exact, validated description of
    the potential it was built with. Defaults exactly match the LES
    upstream defaults so a freshly-constructed
    ``EwaldMultipoleEnergy()`` reproduces ``les.Les()`` numerically
    on the parity-test oracle path.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    sigma: float = Field(default=1.0, gt=0.0)
    dl: float = Field(default=2.0, gt=0.0)
    prefactor: float = Field(default=90.4756, gt=0.0)
    remove_self_interaction: bool = True
    use_epsilon_r_scaling: bool = False


# ---------------------------------------------------------------------------
# EwaldMultipoleEnergy
# ---------------------------------------------------------------------------


class EwaldMultipoleEnergy(BasePotential):
    """LES-style σ-screened multipole electrostatic energy potential.

    Three torch-pme-style kernel methods (:meth:`from_dist`,
    :meth:`lr_from_k_sq`, :meth:`self_contribution`) compute the bare
    pair / reciprocal / self constants. :meth:`forward` dispatches between
    the non-periodic O(N²) realspace path and the periodic reciprocal
    path based on whether ``cell`` is supplied (and is non-degenerate),
    then inlines the non-self-consistent induced response when ``κ``
    and/or ``α`` are present.

    Args:
        sigma: Gaussian charge-smearing length in Å. Default 1.0.
        dl: Reciprocal-space grid resolution in Å (controls k-space
            cutoff ``k_max = 2π/dl``). Default 2.0.
        prefactor: Electrostatic prefactor ``1/(2 ε₀)`` in the project's
            unit system. Default ``90.4756`` (eV·Å·e⁻²).
        remove_self_interaction: If True (default), exclude the
            atom-on-itself Gaussian self-term from both the energy and
            the per-atom potential / field. Required for the
            linear-response path so ``Φ(r_i)`` is the field from
            *other* sources only.
        use_epsilon_r_scaling: If True, apply Clausius-Mossotti-style
            ``ε_r = (Σ_i α_i)/(V·ε₀) + 1`` scaling to the dielectric
            response. Default False (research-feature; off in v0).

    Forward inputs (passed as keyword arguments):
        q: ``(N,)`` or ``(N, n_q)`` per-atom charges; **required**.
        pos: ``(N, 3)`` per-atom positions; **required**.
        cell: ``(3, 3)`` lattice vectors for periodic systems; ``None``
            or zero-determinant ⇒ non-periodic O(N²) path.
        batch: ``(N,)`` int graph membership for batched inputs;
            ``None`` ⇒ single-graph.
        mu: ``(N, 3)`` or ``(N, n_q, 3)`` per-atom dipoles; optional.
        Q: ``(N, 3, 3)`` or ``(N, n_q, 3, 3)`` per-atom quadrupoles;
            optional. Caller is responsible for tracelessness (Flag #2).
        kappa: ``(N,)`` or ``(N, n_q)`` per-atom hardness for the
            induced-charge response; optional. Triggers
            ``q_induced = -κ·Φ`` and ``U_iq = ½ Φ·q_induced``.
        alpha: ``(N,)`` (isotropic) or ``(N, 3, 3)`` (anisotropic)
            per-atom polarizability for the induced-dipole response;
            optional. Triggers ``u_induced = α·E`` and
            ``U_iu = -½ E·u_induced``.
        e_ext: ``(3,)`` external electric field; optional, added to
            the per-atom field before the polarizability response.
        compute_field: If True, populate ``field`` in the output even
            when ``α`` is absent. Default False.

    Forward output:
        Dict with keys ``"pot"`` (per-graph energy ``(B,)``),
        ``"phi"`` (per-atom potential ``(N,)`` or ``(N, n_q)``),
        ``"field"`` (per-atom field ``(N, 3)`` or ``(N, n_q, 3)``),
        ``"q_induced"``, ``"u_induced"``.
    """

    name = "ewald_multipole"
    type = "long_range"

    def __init__(
        self,
        *,
        sigma: float = 1.0,
        dl: float = 2.0,
        prefactor: float = 90.4756,
        remove_self_interaction: bool = True,
        use_epsilon_r_scaling: bool = False,
    ):
        super().__init__()
        self.config = EwaldMultipoleEnergySpec(
            sigma=sigma,
            dl=dl,
            prefactor=prefactor,
            remove_self_interaction=remove_self_interaction,
            use_epsilon_r_scaling=use_epsilon_r_scaling,
        )
        cfg = self.config

        # Promote to attributes for hot-path access.
        self.sigma = cfg.sigma
        self.dl = cfg.dl
        self.prefactor = cfg.prefactor
        self.remove_self_interaction = cfg.remove_self_interaction
        self.use_epsilon_r_scaling = cfg.use_epsilon_r_scaling

        # Derived constants.
        self._a = 1.0 / (cfg.sigma * (2.0**0.5))  # = 1/(σ√2)
        self._sigma_sq_half = cfg.sigma * cfg.sigma / 2.0  # σ²/2
        # k-space cutoff with a tiny relative tolerance: bare `k² ≤ k_max²`
        # introduces a step discontinuity exactly at the boundary; under
        # cell rotation the rounded `nvec @ G @ R.T` lookup of k² flips a
        # few boundary k-vectors on/off relative to the unrotated cell,
        # which breaks SO(3) equivariance of the reciprocal sum and
        # FD-stress smoothness. A 1e-10 relative widening keeps the
        # boundary set deterministically inclusive without altering
        # practical cutoff behavior.
        self._k_sq_max = (2.0 * torch.pi / cfg.dl) ** 2 * (1.0 + 1.0e-10)
        # ``norm_const = prefactor / (2π)`` is the ``1/(4πε₀)`` factor used
        # in the bare per-pair real-space kernels (since
        # ``prefactor = 1/(2ε₀) = 2π · 1/(4πε₀)``). Reciprocal sums use
        # ``prefactor / V`` directly per LES `ewald.py`.
        self._norm_const = cfg.prefactor / (2.0 * torch.pi)

    @classmethod
    def from_spec(cls, spec: EwaldMultipoleEnergySpec) -> "EwaldMultipoleEnergy":
        """Construct from a frozen :class:`EwaldMultipoleEnergySpec` snapshot."""
        return cls(**spec.model_dump())

    # ------------------------------------------------------------------
    # Three torch-pme-style kernel methods
    # ------------------------------------------------------------------

    def from_dist(self, r_ij: torch.Tensor) -> dict[str, torch.Tensor]:
        """Realspace screened-Coulomb multipole kernels at pairwise displacements.

        Computes ``f_qq(r), f_qu(r), f_uu(r), f_Qu(r), f_QQ(r)`` at every
        ``r_ij[i, j] = r[j] - r[i]``, all multiplied by ``norm_const =
        prefactor / (2π)``. Diagonal (``i == j``) entries are zero.

        Used internally by the non-periodic O(N²) path. Mirrors
        ``les.module.make_kernels.make_kernels`` line-by-line.

        Args:
            r_ij: ``(N, N, 3)`` pairwise displacement tensor with the
                convention ``r_ij[i, j] = r[j] - r[i]``.

        Returns:
            Dict with keys ``"f_qq"`` ``(N, N)``, ``"f_qu"`` ``(N, N, 3)``,
            ``"f_uu"`` ``(N, N, 3, 3)``, ``"f_Qu"`` ``(N, N, 3, 3, 3)``,
            ``"f_QQ"`` ``(N, N, 3, 3, 3, 3)``.
        """
        n = r_ij.shape[0]
        device = r_ij.device
        dtype = r_ij.dtype
        a = self._a
        sqrt_pi = math.sqrt(math.pi)
        norm_const = self._norm_const

        eye_n = torch.eye(n, dtype=torch.bool, device=device)
        mask_off = ~eye_n  # (N, N) — True off-diagonal

        r_norm = torch.linalg.norm(r_ij, dim=-1)  # (N, N)

        rinv = torch.zeros_like(r_norm)
        rinv[mask_off] = 1.0 / r_norm[mask_off]

        erf_vals = torch.zeros_like(r_norm)
        erf_vals[mask_off] = torch.special.erf(r_norm[mask_off] * a)

        f_qq = erf_vals * rinv * norm_const  # (N, N)

        rinv2 = rinv * rinv
        rinv3 = rinv2 * rinv
        gauss = torch.exp(-((a * r_norm) ** 2)) * mask_off
        rhat = r_ij * rinv.unsqueeze(-1)
        eye3 = torch.eye(3, dtype=dtype, device=device)

        s1 = erf_vals * rinv3 - (2.0 * a / sqrt_pi) * gauss * rinv2
        s2 = (
            3.0 * erf_vals * rinv3
            - (6.0 * a / sqrt_pi) * gauss * rinv2
            - (4.0 * a**3 / sqrt_pi) * gauss
        )

        rr = rhat[..., :, None] * rhat[..., None, :]  # (N, N, 3, 3)
        f_uu = (s2[:, :, None, None] * rr - s1[:, :, None, None] * eye3[None, None]) * norm_const
        f_qu = s1[..., None] * r_ij * norm_const  # (N, N, 3)

        rinv4 = rinv3 * rinv
        rinv5 = rinv4 * rinv
        s3 = (
            15.0 * erf_vals * rinv4
            - (30.0 * a / sqrt_pi) * gauss * rinv3
            - (20.0 * a**3 / sqrt_pi) * gauss * rinv
            - (8.0 * a**5 / sqrt_pi) * gauss * r_norm
        )
        s4 = (
            105.0 * erf_vals * rinv5
            - (210.0 * a / sqrt_pi) * gauss * rinv4
            - (140.0 * a**3 / sqrt_pi) * gauss * rinv2
            - (56.0 * a**5 / sqrt_pi) * gauss
            - (16.0 * a**7 / sqrt_pi) * gauss * r_norm**2
        )

        rrr = torch.einsum("nmi,nmj,nmk->nmijk", rhat, rhat, rhat)
        term_delta_r = (
            torch.einsum("ab,ijc->ijabc", eye3, rhat)
            + torch.einsum("ac,ijb->ijabc", eye3, rhat)
            + torch.einsum("bc,ija->ijabc", eye3, rhat)
        )
        f_Qu = (
            s3[..., None, None, None] * rrr - (s2 * rinv)[..., None, None, None] * term_delta_r
        ) * norm_const

        rrrr = torch.einsum("ija,ijb,ijc,ijd->ijabcd", rhat, rhat, rhat, rhat)
        term_delta_rr = (
            torch.einsum("ab,ijc,ijd->ijabcd", eye3, rhat, rhat)
            + torch.einsum("ac,ijb,ijd->ijabcd", eye3, rhat, rhat)
            + torch.einsum("ad,ijb,ijc->ijabcd", eye3, rhat, rhat)
            + torch.einsum("bc,ija,ijd->ijabcd", eye3, rhat, rhat)
            + torch.einsum("bd,ija,ijc->ijabcd", eye3, rhat, rhat)
            + torch.einsum("cd,ija,ijb->ijabcd", eye3, rhat, rhat)
        )
        term_delta_delta = (
            torch.einsum("ab,cd->abcd", eye3, eye3)
            + torch.einsum("ac,bd->abcd", eye3, eye3)
            + torch.einsum("ad,bc->abcd", eye3, eye3)
        )[None, None]
        f_QQ = (
            s4[..., None, None, None, None] * rrrr
            - (s3 * rinv)[..., None, None, None, None] * term_delta_rr
            + (s2 * rinv2)[..., None, None, None, None] * term_delta_delta
        ) * norm_const

        return {"f_qq": f_qq, "f_qu": f_qu, "f_uu": f_uu, "f_Qu": f_Qu, "f_QQ": f_QQ}

    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        """Reciprocal-space screened-Coulomb kernel ``exp(-σ²k²/2) / k²``.

        Used internally by the periodic reciprocal path. The k=0 term is
        not handled here (the caller is expected to mask it out before
        invoking this method, per the Yeh-Berkowitz uniform-background
        convention — Flag #3).

        Args:
            k_sq: ``(M,)`` squared k-vector magnitudes (must be > 0).

        Returns:
            ``(M,)`` ``kfac(k) = exp(-σ²k²/2) / k²`` in atomic units of
            inverse-length-squared. The caller multiplies by
            ``prefactor / V`` to get the energy prefactor.
        """
        return torch.exp(-self._sigma_sq_half * k_sq) / k_sq

    def self_contribution(self) -> dict[str, float]:
        """Per-multipole self-correction constants (numerical scalars).

        Returns the five scalars used to subtract / add back the
        atom-on-itself Gaussian contribution from the energy and the
        per-atom field, depending on ``remove_self_interaction`` and the
        path (realspace already excludes ``i == j``; reciprocal includes
        the diagonal in S(k) and must subtract).

        Returns:
            Dict with keys:

            * ``"energy_q"`` — coefficient of ``q²`` in the energy
              self-term: ``1 / (σ (2π)^(3/2)) · prefactor``.
            * ``"phi_q"`` — coefficient of ``q`` in the per-atom
              potential self-term: ``2 / (σ (2π)^(3/2)) · prefactor``
              (twice ``"energy_q"`` because the energy carries ½).
            * ``"energy_u"`` — coefficient of ``‖μ‖²`` in the dipole
              self-energy: ``1 / (3 σ³ (2π)^(3/2)) · prefactor``.
            * ``"energy_Q"`` — coefficient of ``‖Q‖_F²`` in the
              quadrupole self-energy: ``1 / (10 σ⁵ (2π)^(3/2)) ·
              prefactor`` (assumes Q symmetric traceless — Flag #2).
            * ``"field_u"`` — coefficient of ``μ`` in the per-atom
              field self-term: ``(4/(3√π)) · a³ · prefactor / (2π)``,
              ``a = 1/(σ√2)``.
        """
        twopi32 = (2.0 * math.pi) ** 1.5
        sigma = self.sigma
        prefactor = self.prefactor
        return {
            "energy_q": prefactor / (sigma * twopi32),
            "phi_q": 2.0 * prefactor / (sigma * twopi32),
            "energy_u": prefactor / (3.0 * sigma**3 * twopi32),
            "energy_Q": prefactor / (10.0 * sigma**5 * twopi32),
            "field_u": (4.0 / (3.0 * math.sqrt(math.pi))) * self._a**3 * self._norm_const,
        }

    # ------------------------------------------------------------------
    # forward dispatch
    # ------------------------------------------------------------------

    def forward(  # type: ignore[override]
        self,
        data: dict[str, Any] | None = None,
        *,
        q: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        mu: torch.Tensor | None = None,
        Q: torch.Tensor | None = None,
        kappa: torch.Tensor | None = None,
        alpha: torch.Tensor | None = None,
        e_ext: torch.Tensor | None = None,
        compute_field: bool = False,
        kvec_indices: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute total LES electrostatic energy + per-atom Φ, E.

        Dispatches between the realspace (non-periodic) and reciprocal
        (periodic) paths based on the presence of ``cell``. Inlines the
        non-self-consistent induced response when ``κ`` and/or ``α`` are
        passed; the response uses ``Φ`` / ``E`` *with* self-correction
        applied (``remove_self_interaction=True`` default), so the
        induced charge sees only the field from *other* atoms.

        Args:
            q: ``(N,)`` per-atom charges. Required.
            pos: ``(N, 3)`` per-atom positions. Required.
            cell: ``(3, 3)`` lattice vectors (or ``(B, 3, 3)`` batched);
                ``None`` or zero-determinant ⇒ non-periodic realspace path.
            batch: ``(N,)`` int graph membership. ``None`` ⇒ single graph.
            mu: optional ``(N, 3)`` per-atom dipoles.
            Q: optional ``(N, 3, 3)`` per-atom quadrupoles. **Caller
                responsibility**: pass symmetric traceless tensors.
            kappa: optional ``(N,)`` per-atom hardness; triggers
                ``q_induced = -κ·Φ`` and ``U_iq = ½ Φ·q_induced``.
            alpha: optional ``(N,)`` (isotropic) or ``(N, 3, 3)``
                (anisotropic) per-atom polarizability; triggers
                ``u_induced = α·E`` and ``U_iu = -½ E·u_induced``.
            e_ext: optional ``(3,)`` external field.
            compute_field: ignored when ``α`` is set (always computes);
                if no ``α`` is set, controls whether ``field`` in the
                output dict is populated. Default ``False``.

        Returns:
            Dict with keys ``"pot"`` (scalar or ``(B,)``), ``"phi"``
            ``(N,)``, ``"field"`` ``(N, 3)``, ``"q_induced"`` ``(N,)``,
            ``"u_induced"`` ``(N, 3)``.
        """
        if data is not None and q is None:
            q = data.get("q") if isinstance(data, dict) else None
        if data is not None and pos is None:
            pos = data.get("pos") if isinstance(data, dict) else None
        if q is None or pos is None:
            raise ValueError(
                "EwaldMultipoleEnergy.forward requires `q` and `pos` "
                "as keyword arguments (or as `data` keys)."
            )

        n = pos.shape[0]
        if batch is None:
            batch_eff = torch.zeros(n, dtype=torch.long, device=pos.device)
            single_graph = True
        else:
            batch_eff = batch
            single_graph = False

        phi_full = torch.zeros_like(q)
        field_full = torch.zeros(n, 3, dtype=pos.dtype, device=pos.device)
        q_ind_full = torch.zeros_like(q)
        u_ind_full = torch.zeros(n, 3, dtype=pos.dtype, device=pos.device)
        per_graph_pot: list[torch.Tensor] = []

        unique = torch.unique(batch_eff)
        for raw in unique:
            i = int(raw.item())
            mask = batch_eff == raw
            pos_i = pos[mask]
            q_i = q[mask]
            mu_i = mu[mask] if mu is not None else None
            Q_i = Q[mask] if Q is not None else None
            kappa_i = kappa[mask] if kappa is not None else None
            alpha_i = alpha[mask] if alpha is not None else None
            if cell is None:
                cell_i = None
            elif cell.dim() == 2:
                cell_i = cell
            else:
                cell_i = cell[i]

            use_reciprocal = cell_i is not None and torch.linalg.det(cell_i).abs() > 1e-6

            if use_reciprocal:
                sub = self._compute_reciprocal(
                    pos_i,
                    q_i,
                    cell_i,
                    mu_i,
                    Q_i,
                    kappa_i,
                    alpha_i,
                    e_ext,
                    kvec_indices=kvec_indices,
                )
            else:
                sub = self._compute_realspace(pos_i, q_i, mu_i, Q_i, kappa_i, alpha_i, e_ext)

            per_graph_pot.append(sub["pot"])
            phi_full[mask] = sub["phi"]
            field_full[mask] = sub["field"]
            if "q_induced" in sub:
                q_ind_full[mask] = sub["q_induced"]
            if "u_induced" in sub:
                u_ind_full[mask] = sub["u_induced"]

        if single_graph:
            pot_out = per_graph_pot[0]
        else:
            pot_out = torch.stack(per_graph_pot)

        return {
            "pot": pot_out,
            "phi": phi_full,
            "field": field_full,
            "q_induced": q_ind_full,
            "u_induced": u_ind_full,
        }

    # ------------------------------------------------------------------
    # Private path implementations
    # ------------------------------------------------------------------

    def _compute_realspace(
        self,
        pos: torch.Tensor,
        q: torch.Tensor,
        mu: torch.Tensor | None,
        Q: torch.Tensor | None,
        kappa: torch.Tensor | None,
        alpha: torch.Tensor | None,
        e_ext: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Non-periodic O(N²) all-pairs path. Mirrors oracle ``brute_realspace``."""
        n = pos.shape[0]
        r_ij = pos.unsqueeze(0) - pos.unsqueeze(1)  # r_ij[i, j] = r[j] - r[i]
        kernels = self.from_dist(r_ij)
        f_qq = kernels["f_qq"]
        f_qu = kernels["f_qu"]
        f_uu = kernels["f_uu"]
        f_Qu = kernels["f_Qu"]
        f_QQ = kernels["f_QQ"]

        e_phi = torch.einsum("i,ij->j", q, f_qq)
        pot = 0.5 * torch.dot(e_phi, q)

        E_u = torch.zeros(n, 3, dtype=pos.dtype, device=pos.device)
        if mu is not None:
            e_phi_u = torch.einsum("ic,ijc->j", mu, f_qu)
            e_phi = e_phi + e_phi_u
            pot = pot + torch.dot(e_phi_u, q)

            E_u = torch.einsum("ijcd,ic->jd", f_uu, mu)
            pot = pot - 0.5 * torch.einsum("ic,ic->", mu, E_u)

        E_Q = torch.zeros(n, 3, dtype=pos.dtype, device=pos.device)
        if Q is not None:
            e_phi_Q = 0.5 * torch.einsum("iab,ijab->j", Q, f_uu)
            E_Q = 0.5 * torch.einsum("iab,ijabc->jc", Q, f_Qu)
            e_phi = e_phi + e_phi_Q

            pot = pot + torch.dot(q, e_phi_Q)
            pot = pot + 0.125 * torch.einsum("iab,ijabcd,jcd->", Q, f_QQ, Q)
            if mu is not None:
                pot = pot - torch.einsum("ic,ic->", mu, E_Q)

        # remove_self_interaction=False ⇒ realspace must add back the self-terms.
        if not self.remove_self_interaction:
            sc = self.self_contribution()
            pot = pot + (q * q).sum() * sc["energy_q"]
            e_phi = e_phi + q * sc["phi_q"]
            if mu is not None:
                pot = pot + (mu * mu).sum() * sc["energy_u"]
                E_u = E_u - sc["field_u"] * mu
            if Q is not None:
                pot = pot + (Q * Q).sum() * sc["energy_Q"]

        e_field = torch.einsum("i,ijc->jc", q, f_qu)
        if mu is not None:
            e_field = e_field + E_u
        if Q is not None:
            e_field = e_field + E_Q
        if e_ext is not None:
            e_field = e_field + e_ext.unsqueeze(0)

        return self._apply_induced(pot, e_phi, e_field, kappa, alpha)

    def enumerate_kvec_indices(self, cell: torch.Tensor) -> torch.Tensor:
        """Enumerate the integer-triplet array used by the reciprocal path.

        Returns the per-cell ``nvec`` ``(M, 3)`` integer triplet array
        that the production reciprocal path would build for ``cell``,
        already filtered by ``keep = (k² > 0) & (k² ≤ k_sq_max)``. This
        is the canonical "frozen k-grid" snapshot a finite-difference
        validator should pin once on the unperturbed cell and pass back
        through :meth:`_compute_reciprocal` (via ``kvec_indices``) so
        that ``E(cell ± dε)`` is evaluated on a *fixed* integer grid —
        without this, the cutoff ``k² ≤ k_sq_max`` is a step
        discontinuity in cell strain (boundary k-vectors flicker
        on/off under FP rounding) and central FD diverges from
        autograd. Standard PME-stress technique; see LAMMPS
        ``fix_numdiff_virial`` for the canonical statement of the
        problem.

        Production callers should leave ``kvec_indices`` at its default
        (``None``) so the dynamic per-cell enumeration runs; freezing
        the grid is *only* appropriate for FD validation where the
        perturbation is small enough that the keep-set is unchanged
        from the unperturbed cell.

        Args:
            cell: ``(3, 3)`` lattice vectors.

        Returns:
            ``(M, 3)`` integer triplet array (cast to the cell's dtype
            for downstream ``nvec @ G`` matmul).
        """
        device = cell.device
        dtype = cell.dtype
        twopi = 2.0 * math.pi
        cell_inv = torch.linalg.inv(cell)
        G = twopi * cell_inv.T
        norms = torch.linalg.norm(cell, dim=1)
        Nk = [max(1, int(norms[i].item() / self.dl)) for i in range(3)]
        n1 = torch.arange(-Nk[0], Nk[0] + 1, device=device)
        n2 = torch.arange(-Nk[1], Nk[1] + 1, device=device)
        n3 = torch.arange(-Nk[2], Nk[2] + 1, device=device)
        nvec = (
            torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1).reshape(-1, 3).to(dtype)
        )
        kvec = nvec @ G
        k_sq = (kvec * kvec).sum(dim=1)
        keep = (k_sq > 0.0) & (k_sq <= self._k_sq_max)
        return nvec[keep]

    def _compute_reciprocal(
        self,
        pos: torch.Tensor,
        q: torch.Tensor,
        cell: torch.Tensor,
        mu: torch.Tensor | None,
        Q: torch.Tensor | None,
        kappa: torch.Tensor | None,
        alpha: torch.Tensor | None,
        e_ext: torch.Tensor | None,
        kvec_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """3D-periodic reciprocal half-k-sphere path. Mirrors oracle ``brute_reciprocal``.

        Args:
            kvec_indices: optional ``(M, 3)`` precomputed integer triplet
                array (typically from :meth:`enumerate_kvec_indices`
                applied to an unperturbed cell). When supplied, the
                method skips the dynamic enumeration and ``keep``
                filter, and instead projects the supplied ``nvec``
                through ``G(cell)`` directly. This is the canonical
                FD-stress path: it makes ``E(cell)`` differentiable
                w.r.t. cell strain near ``cell₀`` because the discrete
                ``keep`` flip is removed. Production callers must leave
                this ``None``.
        """
        device = pos.device
        dtype = pos.dtype
        twopi = 2.0 * math.pi
        sigma_sq_half = self._sigma_sq_half
        prefactor = self.prefactor

        volume = torch.linalg.det(cell).abs()
        cell_inv = torch.linalg.inv(cell)
        G = twopi * cell_inv.T

        if kvec_indices is None:
            k_sq_max = self._k_sq_max
            norms = torch.linalg.norm(cell, dim=1)
            Nk = [max(1, int(norms[i].item() / self.dl)) for i in range(3)]
            n1 = torch.arange(-Nk[0], Nk[0] + 1, device=device)
            n2 = torch.arange(-Nk[1], Nk[1] + 1, device=device)
            n3 = torch.arange(-Nk[2], Nk[2] + 1, device=device)
            nvec = (
                torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1)
                .reshape(-1, 3)
                .to(dtype)
            )
            kvec = nvec @ G  # (M, 3)
            k_sq = (kvec * kvec).sum(dim=1)
            keep = (k_sq > 0.0) & (k_sq <= k_sq_max)
            kvec = kvec[keep]
            k_sq = k_sq[keep]
            nvec = nvec[keep]
        else:
            # Frozen-grid path: project the supplied integer triplet
            # array through the *current* cell so the kvecs (and the
            # k_sq weighting kfac = exp(-σ² k²/2) / k²) stay smooth in
            # cell strain. No discontinuity; FD agrees with autograd.
            nvec = kvec_indices.to(dtype=dtype, device=device)
            kvec = nvec @ G
            k_sq = (kvec * kvec).sum(dim=1)

        # Half-k-sphere: keep one of every ±k pair, factor=2 on non-axis.
        non_zero = (nvec != 0).to(torch.long)
        first_non_zero = torch.argmax(non_zero, dim=1)
        sign = torch.gather(nvec, 1, first_non_zero.unsqueeze(1)).squeeze(-1)
        hemisphere = (sign > 0) | (nvec == 0).all(dim=1)
        kvec = kvec[hemisphere]
        k_sq = k_sq[hemisphere]
        factors = torch.where(
            (nvec[hemisphere] == 0).all(dim=1),
            torch.tensor(1.0, dtype=dtype, device=device),
            torch.tensor(2.0, dtype=dtype, device=device),
        )

        # Structure factor S(k) — split into real / imag parts.
        k_dot_r = pos @ kvec.T  # (N, M)
        cos_kr = torch.cos(k_dot_r)
        sin_kr = torch.sin(k_dot_r)
        S_real = torch.einsum("i,im->m", q, cos_kr)
        S_imag = torch.einsum("i,im->m", q, sin_kr)

        if mu is not None:
            uk = mu @ kvec.T  # (N, M)
            S_real = S_real - torch.einsum("im,im->m", uk, sin_kr)
            S_imag = S_imag + torch.einsum("im,im->m", uk, cos_kr)

        if Q is not None:
            qk2 = torch.einsum("ma,iab,mb->im", kvec, Q, kvec)
            S_real = S_real - 0.5 * torch.einsum("im,im->m", qk2, cos_kr)
            S_imag = S_imag - 0.5 * torch.einsum("im,im->m", qk2, sin_kr)

        S_sq = S_real * S_real + S_imag * S_imag

        kfac = torch.exp(-sigma_sq_half * k_sq) / k_sq
        pot = (factors * kfac * S_sq).sum() / volume * prefactor

        if self.remove_self_interaction:
            sc = self.self_contribution()
            pot = pot - (q * q).sum() * sc["energy_q"]
            if mu is not None:
                pot = pot - (mu * mu).sum() * sc["energy_u"]
            if Q is not None:
                pot = pot - (Q * Q).sum() * sc["energy_Q"]

        # Per-atom potential at r_i = real part of e^{-i k·r_i} S(k).
        prefactor_arr = factors * 2.0 * kfac / volume * prefactor  # (M,)
        term_real = S_real.unsqueeze(0) * cos_kr + S_imag.unsqueeze(0) * sin_kr  # (N, M)
        e_phi = torch.einsum("m,im->i", prefactor_arr, term_real)

        if self.remove_self_interaction:
            sc = self.self_contribution()
            e_phi = e_phi - q * sc["phi_q"]

        # Per-atom field = imag part of e^{-i k·r_i} S(k) · k.
        term_imag = S_real.unsqueeze(0) * sin_kr - S_imag.unsqueeze(0) * cos_kr  # (N, M)
        e_field = torch.einsum("m,im,mc->ic", prefactor_arr, term_imag, kvec)

        if self.remove_self_interaction and mu is not None:
            sc = self.self_contribution()
            e_field = e_field + sc["field_u"] * mu

        if e_ext is not None:
            e_field = e_field + e_ext.unsqueeze(0)

        return self._apply_induced(pot, e_phi, e_field, kappa, alpha)

    def _apply_induced(
        self,
        pot: torch.Tensor,
        e_phi: torch.Tensor,
        e_field: torch.Tensor,
        kappa: torch.Tensor | None,
        alpha: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Inline non-self-consistent linear response from κ and α."""
        out: dict[str, torch.Tensor] = {"phi": e_phi, "field": e_field}
        if kappa is not None:
            q_induced = -kappa * e_phi
            pot = pot + 0.5 * torch.dot(e_phi, q_induced)
            out["q_induced"] = q_induced
        if alpha is not None:
            if alpha.dim() == 1:
                u_induced = alpha.unsqueeze(-1) * e_field
            else:
                # (N, 3, 3) anisotropic: u_induced[i, c] = α[i, c, d] · E[i, d]
                u_induced = torch.einsum("icd,id->ic", alpha, e_field)
            pot = pot - 0.5 * torch.einsum("ic,ic->", e_field, u_induced)
            out["u_induced"] = u_induced
        out["pot"] = pot
        return out
