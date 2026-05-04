"""Direct permanent atomic multipole prediction + classical electrostatic energy.

Reads pooled atom features (from an Allegro-style edge encoder), predicts
selected atomic moments ``q`` (l=0), ``μ`` (l=1) and ``Θ`` (l=2), applies an
optional total-charge projection, and emits a per-graph electrostatic energy
restricted to the enabled interaction terms. The whole permanent-multipole
prediction surface lives in a single :class:`PermMultipoleHead` — there are
no auxiliary equivariant-readout sub-classes; the l=1 / l=2 paths are
inlined so the head is self-contained and the public surface is one symbol.

This head is intentionally *direct* — charges are produced by an l=0 readout
on the encoder embedding, NOT by extremising an electrostatic Lagrangian
under a charge-conservation constraint. The QEq / CELLI variational variant
(Fuchs/Sanocki/Zavadlav, npj Comput. Mater. **11**, 71 (2025),
https://doi.org/10.1038/s41524-025-01790-4) is a different algorithm with a
KKT solve and per-atom Hirshfeld supervision; it is **out of scope here** and
will live in a separate ``QEqLayer``. Induced response (polarizability) also
belongs to a future ``PolarizableMultipoleLayer``.

Scope:
    * charge head + per-graph total-charge projection (mean-residual
      subtraction; loss gradients flow into the unprojected charges so the
      head learns to predict near-neutral sums on its own)
    * three implemented pair-energy kernels following Stone, *The Theory of
      Intermolecular Forces*, 2nd ed. (Oxford, 2013), §3.3 / Eq. 3.3.5:

        ``qq``  ``q_i q_j / r``
        ``qm``  ``[q_j (R̂·μ_i) − q_i (R̂·μ_j)] / r²``
        ``mm``  ``[μ_i·μ_j − 3 (μ_i·R̂)(μ_j·R̂)] / r³``

      with optional ``erfc`` short-range damping on ``qq`` only;
    * equivariant ``μ`` readout — PaiNN-style scalar-gated l=1 path over
      the encoder's tensor track (slice ``1o`` block, gate per-channel,
      collapse ``u·1o → 1·1o``); output is a 3-vector that rotates as such
    * equivariant ``Θ`` readout — same recipe at l=2 (slice ``2e``,
      collapse ``u·2e → 1·2e``); output is the 5-component traceless
      symmetric basis transforming under Wigner ``D⁽²⁾``
    * remaining energy terms (``qt``, ``mt``, ``tt``) refuse at
      construction with ``NotImplementedError`` until a ``Θ``-tensor /
      Cartesian-form interaction kernel lands.

See ``.claude/specs/multipole-layer.md`` for the full design and limitations.

References for the components actually used here:
    * **Equivariant μ readout** — Schütt, Unke, Gastegger, *PaiNN: Equivariant
      Message Passing for the Prediction of Tensorial Properties and
      Molecular Spectra*, ICML 2021 (arXiv:2102.03150). The
      ``μ_mol = Σ q_i r_i + Σ μ_i`` molecular dipole readout used in the
      training script's auxiliary loss matches PaiNN's ``DipoleMoment``
      head; the per-channel scalar-gated l=1 message structure is the same
      as PaiNN's ``v_j ← φ(s_j) ⊙ v_j`` update.
    * **Pair-centred edge representation** — Musaelian et al.,
      *Nature Communications* **14**, 579 (2023) — the Allegro encoder
      whose tensor track this head consumes.
    * **Multipole interaction kernels** — Stone, *The Theory of
      Intermolecular Forces*, 2nd ed. (2013), §3 — for the higher-order
      energy terms tracked as TODO.

Compatibility note: this module does **not** implement the cited CELLI paper.
Predicted charges here are direct readouts (not Qeq solutions), the total-
charge constraint is a hard projection (not a Lagrange multiplier in a KKT
solve), and the loss in the QM9 training script is energy + molecular-dipole
magnitude (not energy + forces + per-atom Hirshfeld charges). A faithful
CELLI port would be a separate ``QEqLayer``.
"""

from __future__ import annotations

import math
from typing import Sequence

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field, field_validator

from molix import config
from molix.data.types import GraphBatch

# ---------------------------------------------------------------------------
# Term registry
# ---------------------------------------------------------------------------
# The six permanent-multipole pair-energy terms truncated at ``l = 2`` use
# uniform two-letter keys built from the moment symbols
#
#       q  ↔ monopole / atomic charge
#       m  ↔ dipole μ_i (l = 1)
#       t  ↔ quadrupole Θ_i (l = 2)
#
# so the symmetric self-pairs are ``qq``, ``mm``, ``tt`` and the three
# asymmetric cross-pairs are ``qm``, ``qt``, ``mt`` (Stone, §3.3 /
# Eq. 3.3.5).  The implemented set covers the three terms with the
# largest QM9-scale contribution (``qq``, ``qm``, ``mm`` — the
# ``O(r^{-3})`` floor is ~1 meV/pair on organic molecules).  ``qt`` /
# ``mt`` / ``tt`` are reserved keywords so user configs can declare
# intent today and trip a clear boundary; the missing kernels need a
# Cartesian / spherical-tensor multiplication for Θ which is tracked
# separately.

VALID_ENERGY_TERMS: frozenset[str] = frozenset({"qq", "qm", "mm", "qt", "mt", "tt"})
_IMPL_ENERGY_TERMS: frozenset[str] = frozenset({"qq", "qm", "mm"})

# Each term needs a specific subset of moments enabled — ``qm`` for
# instance can't run without both ``charge=True`` and ``dipole=True``.
# Validated at construction time so the failure mode is fail-fast and
# mentions exactly which moment is missing.
_TERM_REQUIREMENTS: dict[str, frozenset[str]] = {
    "qq": frozenset({"charge"}),
    "qm": frozenset({"charge", "dipole"}),
    "mm": frozenset({"dipole"}),
    "qt": frozenset({"charge", "quadrupole"}),
    "mt": frozenset({"dipole", "quadrupole"}),
    "tt": frozenset({"quadrupole"}),
}

VALID_DAMPINGS: frozenset[str] = frozenset({"none", "erfc"})  # "thole" → polarizable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scalar_mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    """``[Linear → SiLU] × len(hidden) → Linear`` (no activation on final layer)."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h, dtype=config.ftype))
        layers.append(nn.SiLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim, dtype=config.ftype))
    return nn.Sequential(*layers)


def _find_irrep_offset(
    tensor_irreps: cue.Irreps,
    target_ir: cue.Irrep,
) -> int:
    """Locate ``target_ir``'s starting offset in an ``ir_mul`` flat layout.

    Returns the offset (in scalar elements) of the first occurrence of
    ``target_ir`` within ``tensor_irreps``. Raises if absent.
    """
    offset = 0
    for mul, ir in tensor_irreps:
        if ir == target_ir:
            return offset
        offset += mul * ir.dim
    raise ValueError(f"tensor_irreps must contain a {target_ir} segment; got {tensor_irreps}.")


# ---------------------------------------------------------------------------
# Pydantic config (mirrors molzoo.AllegroSpec / molpot.heads style)
# ---------------------------------------------------------------------------


class PermMultipoleHeadSpec(BaseModel):
    """Configuration snapshot for :class:`PermMultipoleHead`.

    Mirrors the ``AllegroSpec`` pattern: every constructor argument is
    captured here so that a trained checkpoint carries an exact, validated
    description of the head it was built with. molcfg-driven training
    scripts can construct ``PermMultipoleHeadSpec(**cfg["multipole"])`` and
    pass it straight to ``PermMultipoleHead.from_spec(...)``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    input_dim: int = Field(..., gt=0)
    avg_num_neighbors: float | None = Field(default=None, gt=0.0)
    charge: bool = True
    dipole: bool = False
    quadrupole: bool = False
    energy_terms: tuple[str, ...] = ("qq",)
    cutoff: float | None = Field(default=None, gt=0.0)
    damping: str = "none"
    damping_alpha: float = Field(default=0.2, gt=0.0)
    constrain_total_charge: bool = True
    total_charge_key: str = "total_charge"
    embed_moments: bool = False
    hidden_dim: int = Field(default=128, gt=0)
    coulomb_constant: float = Field(default=14.399645, gt=0.0)
    out_energy_key: str = "energy_es"
    out_charge_key: str = "atomic_charges"
    out_dipole_key: str = "atomic_dipoles"
    out_quadrupole_key: str = "atomic_quadrupoles"

    @field_validator("energy_terms", mode="before")
    @classmethod
    def _coerce_terms(cls, v: object) -> tuple[str, ...]:
        if isinstance(v, str):
            return (v,)
        return tuple(v)  # type: ignore[arg-type]

    @field_validator("energy_terms")
    @classmethod
    def _validate_terms(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        unknown = set(v) - VALID_ENERGY_TERMS
        if unknown:
            raise ValueError(
                f"Unknown energy_terms: {sorted(unknown)}. Valid: {sorted(VALID_ENERGY_TERMS)}."
            )
        return v

    @field_validator("damping")
    @classmethod
    def _validate_damping(cls, v: str) -> str:
        if v not in VALID_DAMPINGS:
            raise ValueError(f"Unknown damping {v!r}; valid: {sorted(VALID_DAMPINGS)}.")
        return v


# ---------------------------------------------------------------------------
# PermMultipoleHead
# ---------------------------------------------------------------------------


class PermMultipoleHead(nn.Module):
    """Direct permanent atomic multipole prediction + electrostatic energy.

    Pipeline::

        edge_features (E, F)
            ─ scatter (1/√⟨N⟩) by source ─→  atom_feats (N, F)
        atom_feats
            ─ q_head ─→  q (N,) ─ project total charge ─→  q
        edge_tensor_features (E, irreps_dim) + edge_features (E, F)
            ─ inlined PaiNN-style l=1 readout ─→  μ (N, 3)   [optional]
            ─ inlined PaiNN-style l=2 readout ─→  Θ (N, 5)   [optional]
        (q, μ, Θ, edge_index, bond_dist, atom.batch)
            ─ enabled energy kernels ─→  E_es (B,)
        (q, μ, atoms.pos, atoms.batch)
            ─ derived ─→  μ_mol (B, 3)         (always when q is predicted)

    The two equivariant moment readouts share one structural recipe — slice
    the ``lℓ`` block of the encoder's tensor track, gate per-channel by an
    l=0 scalar projection of the edge features, collapse ``u·lℓ → 1·lℓ``
    with :class:`cuet.Linear`, scatter to source atoms — so they are
    implemented as one private method (:meth:`_equivariant_moment_readout`)
    invoked twice, not as two separate sub-modules.

    Important — what this is NOT:
        * **Not QEq.** Charges are NOT obtained by extremising
          ``E_es(q) + Σ_i (χ_i q_i + ½ J_i q_i²) + λ(Σ q − Q_tot)``.
          They are direct l=0 predictions of the atom embedding.
          See ``QEqLayer`` (TODO) for the variational variant.
        * **Not polarizable.** Induced moments require self-consistent
          response and live in ``PolarizableMultipoleLayer`` (TODO).
        * **Not periodic.** The energy is computed on the encoder's
          neighbour list (``edges.edge_index`` / ``edges.bond_dist``).
          For periodic long-range, wrap with a future
          ``PeriodicMultipoleEnergy`` (Ewald / PME) module.

    Args:
        input_dim: Per-edge feature dim ``F`` from the encoder.
        avg_num_neighbors: Dataset-wide ⟨|N(i)|⟩ for the edge→atom pool
            normalisation. ``None`` falls back to per-source ``1/√|N(i)|``.
        charge: Predict atomic charges ``q_i``. Default ``True``.
        dipole: Predict atomic dipoles ``μ_i ∈ ℝ^3`` via the inlined
            PaiNN-style l=1 readout. Default ``False``. Requires
            ``tensor_irreps`` and the encoder built with
            ``expose_tensor_track=True``.
        quadrupole: Predict atomic quadrupoles ``Θ_i ∈ ℝ^5`` (traceless
            symmetric basis, ``2e`` irrep ordering) via the inlined l=2
            readout. Default ``False``. Requires ``tensor_irreps`` and
            the encoder built with ``l_max ≥ 2`` and
            ``expose_tensor_track=True``. Output transforms under the
            Wigner ``D⁽²⁾(R)`` representation.
        energy_terms: Which interaction terms to include in the per-graph
            electrostatic energy. v0 implements only ``"qq"``; other terms
            raise ``NotImplementedError`` at construction time even if
            their corresponding moment is enabled. Predicted moments are
            still emitted so they can drive auxiliary losses.
        cutoff: Optional electrostatic cutoff in Å. ``None`` reuses the
            encoder's neighbour list as-is.
        damping: Pair damping for ``qq``. ``"none"`` → bare ``1/r``;
            ``"erfc"`` → ``erfc(α r) / r`` (short-range Ewald form).
            ``"thole"`` is reserved for the polarizable variant.
        damping_alpha: ``α`` for ``erfc`` damping (1/Å).
        constrain_total_charge: If ``True``, project predicted charges so
            each graph sums to ``total_charge_key``. Done by subtracting
            the per-graph mean residual — the simplest unbiased
            projection (gradients still flow into the unprojected
            charges, so the head learns to predict near-neutral sums on
            its own; the projection is a safety net, not a workaround for
            a missing loss term).
        total_charge_key: Per-graph total-charge key under ``graphs``.
            **Required** when ``constrain_total_charge=True``; absent →
            ``KeyError`` at forward. For uniformly-neutral datasets
            (e.g. QM9), inject it via
            :class:`~molix.data.tasks.ConstantLabel` (``value=0.0``).
        embed_moments: Reserved. Re-injecting moment information into the
            encoder is a v1 TODO; passing ``True`` raises today.
        hidden_dim: Hidden width of the per-moment scalar MLP heads.
        coulomb_constant: Numerical Coulomb constant in your unit system.
            Defaults to ``14.399645`` eV·Å·e⁻², matching the
            AtomicDress→eV convention used by ``examples/train_qm9_multipole.py``.
        out_energy_key: TensorDict key for the electrostatic energy.
        out_charge_key: TensorDict key for atomic charges.
        out_dipole_key: TensorDict key for atomic dipoles.
        out_quadrupole_key: TensorDict key for atomic quadrupoles.
        tensor_irreps: Wiring (not in spec): irreps of the encoder's
            tensor-track output. Required when ``dipole=True`` (l=1
            readout) or ``quadrupole=True`` (l=2 readout). Pass
            ``Allegro.tensor_track_irreps`` (encoder must be constructed
            with ``expose_tensor_track=True``, and ``l_max ≥ 2`` if
            quadrupole is on).

    Forward output:
        Dict with the same keys as the per-atom and per-graph writes plus
        ``"molecular_dipole"`` when charges are predicted. When total-charge
        projection is on, the output also contains ``"charge_sum_pre_proj"``
        and ``"charge_sum_post_proj"`` per-graph sums for monitoring.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        avg_num_neighbors: float | None = None,
        charge: bool = True,
        dipole: bool = False,
        quadrupole: bool = False,
        energy_terms: Sequence[str] = ("qq",),
        cutoff: float | None = None,
        damping: str = "none",
        damping_alpha: float = 0.2,
        constrain_total_charge: bool = True,
        total_charge_key: str = "total_charge",
        embed_moments: bool = False,
        hidden_dim: int = 128,
        coulomb_constant: float = 14.399645,
        out_energy_key: str = "energy_es",
        out_charge_key: str = "atomic_charges",
        out_dipole_key: str = "atomic_dipoles",
        out_quadrupole_key: str = "atomic_quadrupoles",
        tensor_irreps: cue.Irreps | None = None,
    ):
        super().__init__()
        # Build (and validate) the spec from the kwargs — single source of
        # truth for the head's configuration.  The spec is frozen, so it
        # also serves as a checkpointable description.
        self.config = PermMultipoleHeadSpec(
            input_dim=input_dim,
            avg_num_neighbors=avg_num_neighbors,
            charge=charge,
            dipole=dipole,
            quadrupole=quadrupole,
            energy_terms=tuple(energy_terms),
            cutoff=cutoff,
            damping=damping,
            damping_alpha=damping_alpha,
            constrain_total_charge=constrain_total_charge,
            total_charge_key=total_charge_key,
            embed_moments=embed_moments,
            hidden_dim=hidden_dim,
            coulomb_constant=coulomb_constant,
            out_energy_key=out_energy_key,
            out_charge_key=out_charge_key,
            out_dipole_key=out_dipole_key,
            out_quadrupole_key=out_quadrupole_key,
        )
        cfg = self.config

        if not (cfg.charge or cfg.dipole or cfg.quadrupole):
            raise ValueError(
                "PermMultipoleHead needs at least one moment enabled "
                "(charge / dipole / quadrupole)."
            )
        unimplemented = set(cfg.energy_terms) - _IMPL_ENERGY_TERMS
        if unimplemented:
            raise NotImplementedError(
                f"energy_terms {sorted(unimplemented)} not implemented; "
                f"available: {sorted(_IMPL_ENERGY_TERMS)}. The Θ-tensor "
                "kernels (qt / mt / tt) are tracked as TODO in the spec."
            )
        # Each term's required moments must all be enabled.  Iterating
        # gives a precise per-term error rather than a generic message.
        moments_on = {
            name
            for name, on in [
                ("charge", cfg.charge),
                ("dipole", cfg.dipole),
                ("quadrupole", cfg.quadrupole),
            ]
            if on
        }
        for term in cfg.energy_terms:
            missing = _TERM_REQUIREMENTS[term] - moments_on
            if missing:
                raise ValueError(
                    f"energy_term {term!r} requires {sorted(missing)} "
                    f"to be enabled (got moments={sorted(moments_on)})."
                )
        if cfg.embed_moments:
            raise NotImplementedError(
                "embed_moments=True is reserved for a v1 release "
                "(re-inject moments into edge_features)."
            )

        # Promote a few flags onto self for hot-path access without
        # touching the Pydantic accessor on every forward.
        self.charge = cfg.charge
        self.dipole = cfg.dipole
        self.quadrupole = cfg.quadrupole
        self.energy_terms = cfg.energy_terms
        self.cutoff = cfg.cutoff
        self.damping = cfg.damping
        self.damping_alpha = cfg.damping_alpha
        self.constrain_total_charge = cfg.constrain_total_charge
        self.total_charge_key = cfg.total_charge_key
        self.coulomb_constant = cfg.coulomb_constant
        self.avg_num_neighbors = cfg.avg_num_neighbors
        self.out_energy_key = cfg.out_energy_key
        self.out_charge_key = cfg.out_charge_key
        self.out_dipole_key = cfg.out_dipole_key
        self.out_quadrupole_key = cfg.out_quadrupole_key

        # Charge readout: l=0 scalar MLP on pooled atom features.
        if cfg.charge:
            self.q_head = _scalar_mlp(cfg.input_dim, [cfg.hidden_dim], 1)

        # Equivariant moment readouts share one structural recipe (see
        # _equivariant_moment_readout).  Both require the encoder's
        # tensor-track irreps and uniform multiplicity ``u``.
        if cfg.dipole or cfg.quadrupole:
            if tensor_irreps is None:
                needs = ", ".join(
                    name
                    for name, on in [("dipole", cfg.dipole), ("quadrupole", cfg.quadrupole)]
                    if on
                )
                raise ValueError(
                    f"{needs} requires tensor_irreps (the encoder's "
                    "tensor-track irreps). Construct your encoder with "
                    "expose_tensor_track=True and pass "
                    "tensor_irreps=encoder.tensor_track_irreps."
                )
            muls = tensor_irreps.muls
            if len(set(muls)) != 1:
                raise ValueError(
                    "PermMultipoleHead requires uniform multiplicity in "
                    f"tensor_irreps; got muls={muls}."
                )
            self._u = int(muls[0])
            self._tensor_irreps = tensor_irreps
            G = tensor_irreps.irrep_class
        else:
            self._u = 0  # unused

        # μ readout (l=1, parity odd): slice 1o block, gate, collapse u·1o → 1·1o.
        if cfg.dipole:
            self._mu_offset = _find_irrep_offset(tensor_irreps, G(1, -1))
            self._mu_size = 3 * self._u
            self.mu_proj = nn.Linear(cfg.input_dim, self._u, dtype=config.ftype)
            self.mu_collapse = cuet.Linear(
                irreps_in=cue.Irreps(G, [(self._u, "1o")]),
                irreps_out=cue.Irreps(G, [(1, "1o")]),
                layout=cue.ir_mul,
            )

        # Θ readout (l=2, parity even): slice 2e block, gate, collapse u·2e → 1·2e.
        if cfg.quadrupole:
            try:
                self._theta_offset = _find_irrep_offset(tensor_irreps, G(2, +1))
            except ValueError as e:
                raise ValueError(
                    f"quadrupole=True needs a 2e segment in tensor_irreps; "
                    f"build the encoder with l_max>=2. ({e})"
                ) from None
            self._theta_size = 5 * self._u
            self.theta_proj = nn.Linear(cfg.input_dim, self._u, dtype=config.ftype)
            self.theta_collapse = cuet.Linear(
                irreps_in=cue.Irreps(G, [(self._u, "2e")]),
                irreps_out=cue.Irreps(G, [(1, "2e")]),
                layout=cue.ir_mul,
            )

    @classmethod
    def from_spec(
        cls,
        spec: PermMultipoleHeadSpec,
        *,
        tensor_irreps: cue.Irreps | None = None,
    ) -> "PermMultipoleHead":
        """Construct from a spec.

        ``tensor_irreps`` is wiring (depends on the encoder you compose with),
        not a hyperparameter — it is *not* in the spec and must be supplied
        here when ``spec.dipole=True`` or ``spec.quadrupole=True``. Pass
        ``encoder.tensor_track_irreps`` from an Allegro built with
        ``expose_tensor_track=True`` (and ``l_max >= 2`` if quadrupole is on).
        """
        return cls(**spec.model_dump(), tensor_irreps=tensor_irreps)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, batch: GraphBatch) -> dict[str, torch.Tensor]:
        edge_features = batch["edges", "edge_features"]  # (E, F)
        edge_index = batch["edges", "edge_index"]  # (E, 2)
        bond_dist = batch["edges", "bond_dist"]  # (E,)
        # ``bond_diff`` is required by qm / mm (they project moments onto
        # the unit edge vector R̂ = bond_diff / r).  qq alone doesn't
        # need it, but reading it unconditionally keeps the forward path
        # branch-free and bond_diff is part of the standard edge schema.
        bond_diff = batch["edges", "bond_diff"]  # (E, 3)
        atom_batch = batch["atoms", "batch"]  # (N,)
        n_nodes: int = int(batch["atoms", "Z"].shape[0])
        n_graphs: int = int(batch["graphs"].batch_size[0])
        device = edge_features.device
        dtype = edge_features.dtype

        # 1. Pool edges → atoms (only needed for the l=0 charge head; the
        #    equivariant moment readouts do their own source-side scatter
        #    inside _equivariant_moment_readout).
        atom_feats = (
            self._scatter_mean_to_atoms(edge_features, edge_index, n_nodes) if self.charge else None
        )  # (N, F)

        out: dict[str, torch.Tensor] = {}

        # 2. Heads + per-graph total-charge projection.
        q: torch.Tensor | None = None
        if self.charge:
            q_raw = self.q_head(atom_feats).squeeze(-1)  # (N,)
            if self.constrain_total_charge:
                q, sum_pre, sum_post = self._project_total_charge(
                    q_raw, atom_batch, batch, n_graphs
                )
                out["charge_sum_pre_proj"] = sum_pre
                out["charge_sum_post_proj"] = sum_post
            else:
                q = q_raw
            batch[("atoms", self.out_charge_key)] = q
            out[self.out_charge_key] = q

        mu: torch.Tensor | None = None
        if self.dipole or self.quadrupole:
            tensor_feats = batch["edges", "edge_tensor_features"]  # (E, irreps_dim)
            src = edge_index[:, 0]
            if self.dipole:
                mu = self._equivariant_moment_readout(
                    tensor_feats=tensor_feats,
                    scalar_feats=edge_features,
                    src=src,
                    n_nodes=n_nodes,
                    block_offset=self._mu_offset,
                    block_size=self._mu_size,
                    out_dim=3,
                    scalar_proj=self.mu_proj,
                    collapse=self.mu_collapse,
                )  # (N, 3)
                batch[("atoms", self.out_dipole_key)] = mu
                out[self.out_dipole_key] = mu
            if self.quadrupole:
                theta = self._equivariant_moment_readout(
                    tensor_feats=tensor_feats,
                    scalar_feats=edge_features,
                    src=src,
                    n_nodes=n_nodes,
                    block_offset=self._theta_offset,
                    block_size=self._theta_size,
                    out_dim=5,
                    scalar_proj=self.theta_proj,
                    collapse=self.theta_collapse,
                )  # (N, 5)
                batch[("atoms", self.out_quadrupole_key)] = theta
                out[self.out_quadrupole_key] = theta

        # 3. Molecular dipole (always derived when charges exist).
        if q is not None:
            mu_mol = self._molecular_dipole(
                q, mu, batch["atoms", "pos"], atom_batch, n_graphs
            )  # (B, 3)
            batch[("graphs", "molecular_dipole")] = mu_mol
            out["molecular_dipole"] = mu_mol

        # 4. Enabled energy terms.  Each kernel returns a per-graph
        #    ``(B,)`` tensor; the dispatch keeps them additive so a
        #    profile can request any subset of {qq, qm, mm}.  Required
        #    moments are guaranteed by __init__ via _TERM_REQUIREMENTS.
        energy = torch.zeros(n_graphs, dtype=dtype, device=device)
        if "qq" in self.energy_terms:
            energy = energy + self._coulomb_qq(q, edge_index, bond_dist, atom_batch, n_graphs)
        if "qm" in self.energy_terms:
            energy = energy + self._coulomb_qm(
                q,
                mu,
                edge_index,
                bond_diff,
                bond_dist,
                atom_batch,
                n_graphs,
            )
        if "mm" in self.energy_terms:
            energy = energy + self._coulomb_mm(
                mu,
                edge_index,
                bond_diff,
                bond_dist,
                atom_batch,
                n_graphs,
            )
        # qt / mt / tt rejected in __init__; this dispatch grows when
        # the Θ-tensor interaction kernels land.
        batch[("graphs", self.out_energy_key)] = energy
        out[self.out_energy_key] = energy
        return out

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _equivariant_moment_readout(
        self,
        *,
        tensor_feats: torch.Tensor,  # (E, irreps_dim) ir_mul layout
        scalar_feats: torch.Tensor,  # (E, F)
        src: torch.Tensor,  # (E,) edge source indices
        n_nodes: int,
        block_offset: int,  # start of the lℓ block in tensor_feats
        block_size: int,  # = (2ℓ+1) * u
        out_dim: int,  # = 2ℓ + 1
        scalar_proj: nn.Module,  # F → u  (l=0, rotation-invariant)
        collapse: nn.Module,  # cuet.Linear(u·lℓ → 1·lℓ)
    ) -> torch.Tensor:
        r"""Inlined PaiNN-style scalar-gated lℓ readout.

        Pipeline (per edge ``ij``)::

            v_ij^{ℓ} = slice(V_ij, lℓ).reshape(u, 2ℓ+1)   ∈ ℝ^{u·(2ℓ+1)}
            gate_ij  = scalar_proj(s_ij)                  ∈ ℝ^u            l=0
            gated_ij = gate_ij ⊙ v_ij^{ℓ}                 ∈ ℝ^{u·(2ℓ+1)}   lℓ
            m_ij     = collapse(gated_ij)                                  lℓ
            m_i      = (1/√⟨|N|⟩) · Σ_{j: src=i} m_ij     ∈ ℝ^{2ℓ+1}

        Used by both the μ (ℓ=1) and Θ (ℓ=2) paths — the only differences
        are ``block_offset / block_size / out_dim`` and which weights are
        registered as ``self.mu_*`` vs ``self.theta_*``.

        Strictly SO(3)-equivariant under Wigner ``Dˡ(R)`` because every step
        composes equivariant primitives:
            * scalar gating is l=0 (rotation-invariant).
            * channel-wise multiplication of an l=0 scalar with an lℓ
              vector lives in lℓ.
            * ``cuet.Linear`` between two ``mul × lℓ`` channel
              multiplicities is the equivariant linear map for that irrep.
            * scatter-sum is rotation-invariant in the index reduction.
        """
        E = tensor_feats.shape[0]
        u = self._u
        # Slice the lℓ block in ir_mul layout: (E, (2ℓ+1)*u) flattened
        # along the (component-outer, channel-inner) axes.
        v_flat = tensor_feats[:, block_offset : block_offset + block_size]
        v_l = v_flat.reshape(E, out_dim, u).transpose(1, 2)  # (E, u, 2ℓ+1)

        gate = scalar_proj(scalar_feats)  # (E, u)
        gated = gate.unsqueeze(-1) * v_l  # (E, u, 2ℓ+1)

        # Repack to ir_mul layout (2ℓ+1, u) for cuet.Linear.
        gated_ir_mul = gated.transpose(1, 2).reshape(E, block_size)
        edge_out = collapse(gated_ir_mul)  # (E, 2ℓ+1)

        atom_out = torch.zeros(
            n_nodes,
            out_dim,
            dtype=edge_out.dtype,
            device=edge_out.device,
        )
        atom_out.scatter_add_(
            0,
            src.unsqueeze(-1).expand_as(edge_out),
            edge_out,
        )
        if self.avg_num_neighbors is not None:
            atom_out = atom_out / math.sqrt(self.avg_num_neighbors)
        return atom_out

    def _scatter_mean_to_atoms(
        self,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        src = edge_index[:, 0]
        atom_feats = torch.zeros(
            n_nodes,
            edge_feats.shape[-1],
            dtype=edge_feats.dtype,
            device=edge_feats.device,
        )
        atom_feats.scatter_add_(0, src.unsqueeze(-1).expand_as(edge_feats), edge_feats)
        if self.avg_num_neighbors is not None:
            return atom_feats / math.sqrt(self.avg_num_neighbors)
        count = torch.zeros(n_nodes, dtype=edge_feats.dtype, device=edge_feats.device)
        count.scatter_add_(
            0, src, torch.ones(src.shape[0], dtype=edge_feats.dtype, device=src.device)
        )
        return atom_feats / count.clamp(min=1.0).sqrt().unsqueeze(-1)

    def _project_total_charge(
        self,
        q: torch.Tensor,
        atom_batch: torch.Tensor,
        batch: GraphBatch,
        n_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_target = batch["graphs", self.total_charge_key].to(q.dtype)
        n_atoms_per_graph = torch.zeros(n_graphs, dtype=q.dtype, device=q.device)
        n_atoms_per_graph.scatter_add_(0, atom_batch, torch.ones_like(q))
        sum_q = torch.zeros(n_graphs, dtype=q.dtype, device=q.device)
        sum_q.scatter_add_(0, atom_batch, q)
        delta = (q_target - sum_q) / n_atoms_per_graph.clamp(min=1.0)
        q_proj = q + delta[atom_batch]
        sum_q_post = torch.zeros(n_graphs, dtype=q.dtype, device=q.device)
        sum_q_post.scatter_add_(0, atom_batch, q_proj)
        return q_proj, sum_q, sum_q_post

    def _molecular_dipole(
        self,
        q: torch.Tensor,
        mu: torch.Tensor | None,
        pos: torch.Tensor,
        atom_batch: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        # μ_mol = Σ_i q_i r_i  (+ Σ_i μ_i, if atomic dipoles are predicted).
        # For neutral systems μ_mol is origin-independent — projection
        # ensures Σ q_i = 0, so the value matches the experimental dipole
        # up to the chosen origin convention.
        per_atom = q.unsqueeze(-1) * pos  # (N, 3)
        if mu is not None:
            per_atom = per_atom + mu
        mu_mol = torch.zeros(n_graphs, 3, dtype=per_atom.dtype, device=per_atom.device)
        mu_mol.scatter_add_(0, atom_batch.unsqueeze(-1).expand_as(per_atom), per_atom)
        return mu_mol

    def _coulomb_qq(
        self,
        q: torch.Tensor,
        edge_index: torch.Tensor,
        bond_dist: torch.Tensor,
        atom_batch: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        """``Σ_{i<j} q_i q_j / r_{ij}`` (Stone Eq. 3.3.5, ℓ=0,0 term).

        Edges are bidirectional → each unordered pair appears twice as
        ordered (i, j) and (j, i); the ``0.5`` factor halves the doubled
        sum.  Optional ``erfc`` short-range damping replaces ``1/r`` with
        ``erfc(αr)/r`` (Stone §6.7).
        """
        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        if self.cutoff is not None:
            mask = bond_dist <= self.cutoff
            if not mask.all():
                src = src[mask]
                dst = dst[mask]
                bond_dist = bond_dist[mask]
        inv_r = self._screened_inv_r(bond_dist)
        e_pair = 0.5 * self.coulomb_constant * q[src] * q[dst] * inv_r
        return self._scatter_pair_to_graph(
            e_pair,
            src,
            atom_batch,
            n_graphs,
        )

    def _coulomb_qm(
        self,
        q: torch.Tensor,
        mu: torch.Tensor,
        edge_index: torch.Tensor,
        bond_diff: torch.Tensor,
        bond_dist: torch.Tensor,
        atom_batch: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        r"""``Σ_{i<j} [q_j (R̂·μ_i) − q_i (R̂·μ_j)] / r²`` (Stone, ℓ=0,1 term).

        Convention: ``R̂_{ij} = (r_j − r_i) / r_{ij} = bond_diff / r``
        (molnex edge convention, CLAUDE.md).  The ordered-edge kernel
        ``K(i,j) = (q_j (R̂·μ_i) − q_i (R̂·μ_j)) / r²`` is sign-symmetric
        under swap ``(i,j)↔(j,i)`` because both ``q×μ`` swap *and* ``R̂``
        flips, so a bidirectional edge list double-counts and the
        ``0.5`` factor halves it back to the unordered-pair sum.

        Reference: A. J. Stone, *The Theory of Intermolecular Forces*,
        2nd ed. (Oxford, 2013), §3.3 (charge-dipole interaction).
        """
        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        if self.cutoff is not None:
            mask = bond_dist <= self.cutoff
            if not mask.all():
                src = src[mask]
                dst = dst[mask]
                bond_dist = bond_dist[mask]
                bond_diff = bond_diff[mask]
        inv_r = bond_dist + 1e-12
        r_hat = bond_diff / inv_r.unsqueeze(-1)  # (E, 3)
        # (R̂ · μ_i) and (R̂ · μ_j)
        rhat_dot_mu_src = (r_hat * mu[src]).sum(dim=-1)  # (E,)
        rhat_dot_mu_dst = (r_hat * mu[dst]).sum(dim=-1)  # (E,)
        e_pair = (
            0.5
            * self.coulomb_constant
            * (q[dst] * rhat_dot_mu_src - q[src] * rhat_dot_mu_dst)
            / (bond_dist * bond_dist + 1e-24)
        )
        return self._scatter_pair_to_graph(
            e_pair,
            src,
            atom_batch,
            n_graphs,
        )

    def _coulomb_mm(
        self,
        mu: torch.Tensor,
        edge_index: torch.Tensor,
        bond_diff: torch.Tensor,
        bond_dist: torch.Tensor,
        atom_batch: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        r"""``Σ_{i<j} [μ_i·μ_j − 3(μ_i·R̂)(μ_j·R̂)] / r³`` (Stone, ℓ=1,1 term).

        Symmetric under ``(i,j)↔(j,i)`` (both ``μ_i·μ_j`` and the
        ``(μ·R̂)(μ·R̂)`` product are invariant under swap *and* under
        ``R̂ → −R̂``), so the ordered-edge kernel matches on both
        directions and the ``0.5`` factor recovers the unordered sum
        from the bidirectional edge list.

        Reference: Stone 2013 §3.3, dipole-dipole interaction (the
        textbook ``T_{ab}^{(1,1)} = (1 − 3 R̂R̂)/r³`` form).
        """
        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        if self.cutoff is not None:
            mask = bond_dist <= self.cutoff
            if not mask.all():
                src = src[mask]
                dst = dst[mask]
                bond_dist = bond_dist[mask]
                bond_diff = bond_diff[mask]
        inv_r = bond_dist + 1e-12
        r_hat = bond_diff / inv_r.unsqueeze(-1)  # (E, 3)
        mu_dot_mu = (mu[src] * mu[dst]).sum(dim=-1)  # (E,)
        rhat_dot_mu_src = (r_hat * mu[src]).sum(dim=-1)  # (E,)
        rhat_dot_mu_dst = (r_hat * mu[dst]).sum(dim=-1)  # (E,)
        r3 = bond_dist * bond_dist * bond_dist + 1e-36
        e_pair = (
            0.5 * self.coulomb_constant * (mu_dot_mu - 3.0 * rhat_dot_mu_src * rhat_dot_mu_dst) / r3
        )
        return self._scatter_pair_to_graph(
            e_pair,
            src,
            atom_batch,
            n_graphs,
        )

    @staticmethod
    def _scatter_pair_to_graph(
        e_pair: torch.Tensor,
        src: torch.Tensor,
        atom_batch: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        """Sum per-edge contributions into per-graph energies."""
        graph_idx = atom_batch[src]
        energy = torch.zeros(n_graphs, dtype=e_pair.dtype, device=e_pair.device)
        energy.scatter_add_(0, graph_idx, e_pair)
        return energy

    def _screened_inv_r(self, r: torch.Tensor) -> torch.Tensor:
        # r already excludes self-pairs (the encoder's neighbour list does
        # not emit i==i), so the small ε is purely numerical hygiene.
        if self.damping == "none":
            return 1.0 / (r + 1e-12)
        if self.damping == "erfc":
            return torch.special.erfc(self.damping_alpha * r) / (r + 1e-12)
        raise NotImplementedError(self.damping)
