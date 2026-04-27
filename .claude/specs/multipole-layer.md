# PermMultipoleHead

| Field | Value |
|-------|-------|
| Slug | `multipole-layer` |
| Package | `molpot` (head) |
| Module | `molpot.heads.multipole` |
| Entry point | `PermMultipoleHead` (config `PermMultipoleHeadSpec`) |
| Status | draft |
| Owning agent | `mn-impl` |
| Reference (μ readout) | Schütt, Unke, Gastegger, *PaiNN: Equivariant Message Passing for the Prediction of Tensorial Properties and Molecular Spectra*, ICML 2021 — https://arxiv.org/abs/2102.03150 |
| Reference (encoder) | Musaelian et al., *Allegro*, *Nat. Commun.* **14**, 579 (2023) |
| Out-of-scope reference | Fuchs/Sanocki/Zavadlav, *CELLI*, npj Comput. Mater. **11**, 71 (2025) — https://doi.org/10.1038/s41524-025-01790-4 (Qeq + KKT solve + Hirshfeld supervision; **not** what this layer implements; will live in a separate `QEqLayer`) |

## Problem

Allegro emits scalar `("edges", "edge_features")` and `EdgeEnergyHead` aggregates
those into a per-graph energy. That gives a serviceable potential but no
physical access to the underlying charge density: we can't predict atomic
moments, can't supervise on electrostatic potential / molecular dipole, and
can't compose with classical multipole-multipole interaction kernels.

The `PermMultipoleHead` is a **PaiNN-style direct multipole readout** over the
Allegro encoder:

* charges ``q_i`` come from a per-atom scalar head on Allegro's pooled scalar
  features, with a hard mean-residual projection onto ``Σ_i q_i = Q_tot``;
* atomic dipoles ``μ_i`` come from an inlined PaiNN-style scalar-gated `l=1`
  readout over Allegro's tensor track (``v_j ← φ(s_j) ⊙ v_j``, regression-
  tested for SO(3) equivariance);
* atomic quadrupoles ``Θ_i`` come from the same recipe at `l=2` (slice the
  `2e` block, gate per-channel scalars, collapse `u·2e → 1·2e`), output
  transforming under Wigner ``D⁽²⁾``;
* the molecular dipole ``μ_mol = Σ_i q_i r_i + Σ_i μ_i`` (PaiNN's
  ``DipoleMoment`` head) is emitted automatically and can be supervised
  against QM9's `mu` magnitude;
* electrostatic energy enters the total potential through ``½ Σ k_C q_i q_j /
  r_{ij}`` (and, in v1, higher tensor terms from Stone §3);
* downstream MM force fields can consume the predicted ``(q, μ, Θ)`` directly
  as transferable atomic parameters.

This layer is **direct** in the same sense as PaiNN: heads → moments → energy.
The Qeq variational route (CELLI, Fuchs et al. 2025) — solving a per-atom KKT
system under ``1ᵀQ = Q_tot`` and supervising against per-atom Hirshfeld
charges — is a different physical model and a different supervision regime;
it is explicitly out of scope here and will live in a separate future
`QEqLayer`. Polarizable response (induced dipoles, self-consistent Thole
damping) and periodic Ewald / PME are likewise separate concerns.

## Non-goals

* **Charge equilibration (Qeq / CELLI).** See header table & §Reference for
  the full statement; the short version is "different code path, future
  `QEqLayer`."
* **Polarizable response.** Permanent + induced dipoles need a self-consistent
  loop; goes in a future `PolarizableMultipoleHead`. The reserved
  `damping="thole"` value belongs there.
* **Periodic Ewald / PME long-range backends.** This layer reuses the
  encoder's neighbour list. Periodic systems get a future
  `PeriodicMultipoleEnergy` wrapper.
* **Higher-order interaction kernels.** Three of the six pair terms
  through `l = 2` are implemented: `qq`, `qm`, `mm` (Stone, *Theory of
  Intermolecular Forces*, 2nd ed., 2013, §3.3 / Eq. 3.3.5):

  * `qq` — `q_a q_b / r`
  * `qm` — `[q_b (R̂·μ_a) − q_a (R̂·μ_b)] / r²`
  * `mm` — `[μ_a·μ_b − 3 (μ_a·R̂)(μ_b·R̂)] / r³`

  The remaining three (`qt`, `mt`, `tt`) require a Cartesian /
  spherical-tensor multiplication on `Θ` and are recognised keywords
  that raise `NotImplementedError` at construction time so user configs
  can declare intent today and trip a clear boundary.

  Each term is gated by `_TERM_REQUIREMENTS` — `qm` requires both
  `charge=True` AND `dipole=True`, `mm` requires `dipole=True`, etc.
  Numerical-validation tests in
  `tests/test_molpot/test_heads/test_multipole_energy_kernels.py` check
  the kernels against hand-derived 2-atom values + composed-pipeline
  translation/rotation invariance.
* **Equivariant `Θ` head — implemented.** Quadrupole prediction is the exact
  `l=2` analogue of the inlined `μ` readout: slice the `2e` block of the
  encoder's tensor track, scalar-gate per-channel, and collapse `u·2e → 1·2e`
  with `cuet.Linear`. Output transforms under the Wigner `D⁽²⁾(R)`
  representation; SO(3) equivariance is regression-tested in the composed
  pipeline at `tests/test_molpot/test_heads/test_multipole_symmetry.py`.
  Requires the encoder to expose `("edges","edge_tensor_features")` AND be
  built with `l_max >= 2` so the `2e` block survives.
* **Atomic-multipole supervision.** QM9 lacks atomic-level labels; ESP-grid
  loss and Hirshfeld / DMA / GDMA targets are deferred.
* **`embed_moments=True`** (re-injecting predicted moments into the encoder's
  edge features). Reserved; raises today.

## Public surface

### Constructor (`PermMultipoleHead.__init__` / `PermMultipoleHeadSpec`)

```python
PermMultipoleHead(
    *,
    input_dim: int,                              # encoder feature dim F
    avg_num_neighbors: float | None = None,
    charge: bool = True,
    dipole: bool = False,
    quadrupole: bool = False,
    energy_terms: tuple[str, ...] = ("qq",),
    cutoff: float | None = None,
    damping: str = "none",                       # "none" | "erfc"
    damping_alpha: float = 0.2,
    constrain_total_charge: bool = True,
    total_charge_key: str = "total_charge",
    embed_moments: bool = False,                 # raises in v0
    hidden_dim: int = 128,
    coulomb_constant: float = 14.399645,         # eV · Å · e^-2
    out_energy_key: str = "energy_es",
    out_charge_key: str = "atomic_charges",
    out_dipole_key: str = "atomic_dipoles",
    out_quadrupole_key: str = "atomic_quadrupoles",
    tensor_irreps: cue.Irreps | None = None,    # required when dipole=True or quadrupole=True
)
```

When `dipole=True` or `quadrupole=True`, the head registers the matching
inlined PaiNN-style readout over the encoder's tensor track:

* `dipole=True` — scalar-gated `l=1` path consuming the `1o` block, output
  is a 3-vector that rotates as one.
* `quadrupole=True` — same recipe at `l=2`, consuming the `2e` block,
  output is the 5-component traceless symmetric basis transforming under
  Wigner `D⁽²⁾`.

Both share the private :meth:`PermMultipoleHead._equivariant_moment_readout`
implementation — there are deliberately *no* separate `EquivariantMuHead` /
`EquivariantThetaHead` sub-classes, the entire permanent-multipole surface
is a single :class:`PermMultipoleHead`. Either path requires the encoder
to expose ``("edges","edge_tensor_features")`` (e.g. ``Allegro(...,
expose_tensor_track=True)``) and the caller to pass
``tensor_irreps=encoder.tensor_track_irreps``. ``quadrupole=True``
additionally requires the encoder's ``l_max >= 2`` so the `2e` block
survives the last-layer pruning. SO(3) equivariance for both readouts is
regression-tested in
``tests/test_molpot/test_heads/test_multipole_symmetry.py``.

Moment prediction and energy-term inclusion are **independent axes**:

```python
# Predict q + μ + Θ but only spend FLOPs on q-q electrostatics
PermMultipoleHead(input_dim=128, dipole=True, quadrupole=True, energy_terms=("qq",))
```

Construction is fail-fast: unknown energy terms / dampings raise `ValueError`;
unimplemented terms raise `NotImplementedError`; `"qq"` without `charge=True`
is rejected.

A `PermMultipoleHeadSpec(BaseModel)` snapshot lives on `self.config` after
construction, mirroring `AllegroSpec`. Intended use from molcfg-driven
training scripts:

```python
encoder = Allegro(..., expose_tensor_track=cfg["multipole"]["dipole"])

spec = PermMultipoleHeadSpec(input_dim=cfg["num_scalar_features"], **cfg["multipole"])
layer = PermMultipoleHead.from_spec(
    spec,
    tensor_irreps=encoder.tensor_track_irreps if spec.dipole else None,
)
```

**`tensor_irreps` is intentionally NOT in the spec.** It's a wiring parameter
(depends on the encoder you compose with), not a hyperparameter; ``cue.Irreps``
also doesn't round-trip cleanly through Pydantic JSON. ``from_spec`` therefore
takes ``tensor_irreps`` as an extra kwarg that the caller must re-supply at
load time from the freshly-constructed encoder.

### Forward

`forward(batch: GraphBatch) -> dict[str, Tensor]`. Mutates `batch` in place
and returns a dict containing the same writes plus diagnostics.

### Composition pattern

```
batch
 └─ Allegro                      (writes ("edges","edge_features"))
 └─ EdgeEnergyHead               (reads it → ("graphs", "energy"))
 └─ PermMultipoleHead               (reads same edges →
                                   atomic moments + ("graphs","energy_es"))
loss = mse(energy + energy_es, U0)  +  λ_μ |μ_mol| MAE
```

## Data contract

### Inputs (consumed from `GraphBatch`)

| Path | Shape | Source |
|------|-------|--------|
| `("edges", "edge_features")` | `(E, F)` | encoder (Allegro) |
| `("edges", "edge_index")` | `(E, 2)` int | NeighborList; `[:,0]=src`, `[:,1]=tgt` |
| `("edges", "bond_dist")` | `(E,)` | NeighborList |
| `("atoms", "Z")` | `(N,)` int | DataModule |
| `("atoms", "pos")` | `(N, 3)` | DataModule |
| `("atoms", "batch")` | `(N,)` int | collate_molecules |
| `("graphs", total_charge_key)` | `(B,)` | dataset / pipeline — **required** when `constrain_total_charge=True`; absent → `KeyError`. For uniformly-neutral datasets, inject via `ConstantLabel(key="total_charge", value=0.0)`. |

### Outputs (written into `batch` + returned dict)

| Path | Shape | When |
|------|-------|------|
| `("atoms", out_charge_key)` | `(N,)` | `charge=True` |
| `("atoms", out_dipole_key)` | `(N, 3)` | `dipole=True` |
| `("atoms", out_quadrupole_key)` | `(N, 5)` | `quadrupole=True` |
| `("graphs", out_energy_key)` | `(B,)` | always |
| `("graphs", "molecular_dipole")` | `(B, 3)` | `charge=True` |
| return dict only: `"charge_sum_pre_proj"` | `(B,)` | `constrain_total_charge=True` |
| return dict only: `"charge_sum_post_proj"` | `(B,)` | `constrain_total_charge=True` |

### Internal pipeline

```
edge_features (E, F)
    ── scatter_add by source ──── atom_feats (N, F)     [+ 1/√⟨|N|⟩ rescale]

atom_feats ── q_head:   Linear→SiLU→Linear → squeeze ── q_raw (N,)
              ── if constrain_total_charge:
                   δ_g = (Q_tot,g − Σ q_raw,i in g) / |g|
                   q_i ← q_raw,i + δ_{g(i)}
              → q (N,)

edge_tensor_features (E, irreps_dim)   [from Allegro(expose_tensor_track=True)]
edge_features        (E, F)
    ── inlined PaiNN-style scalar-gated l=1 readout:
         gate_ij  = scalar_proj(s_ij)              ∈ ℝ^u            l=0
         v_ij^{1} = slice(V_ij, l=1).reshape(u,3)  ∈ ℝ^{u·3}        l=1
         gated_ij = gate_ij ⊙ v_ij^{1}             ∈ ℝ^{u·3}        l=1
         μ_ij     = cuet.Linear(u·1o → 1·1o)(gated_ij)              l=1
         μ_i      = (1/√⟨|N|⟩) · Σ_{j: src=i} μ_ij  ∈ ℝ^3
    → μ (N, 3)                                              [equivariant ✅]

edge_tensor_features (E, irreps_dim)   [from Allegro(expose_tensor_track=True, l_max>=2)]
edge_features        (E, F)
    ── inlined PaiNN-style scalar-gated l=2 readout:
         gate_ij  = scalar_proj(s_ij)              ∈ ℝ^u            l=0
         v_ij^{2} = slice(V_ij, l=2).reshape(u,5)  ∈ ℝ^{u·5}        l=2
         gated_ij = gate_ij ⊙ v_ij^{2}             ∈ ℝ^{u·5}        l=2
         Θ_ij     = cuet.Linear(u·2e → 1·2e)(gated_ij)              l=2
         Θ_i      = (1/√⟨|N|⟩) · Σ_{j: src=i} Θ_ij  ∈ ℝ^5
    → Θ (N, 5)                                              [equivariant ✅]

(q, μ, pos, batch)        ─→ μ_mol = Σ_i (q_i r_i + μ_i)         (B, 3)
(q, edge_index, bond_dist) ─→ E_qq = ½ Σ_edges k_C q_i q_j inv_r(r_ij)   (B,)
                                where inv_r = 1/r  or  erfc(α r)/r
```

The bidirectional encoder neighbour list yields the ordered pair sum, hence
the `½` for unordered Coulomb. `cutoff=None` reuses the encoder's
neighbour list as-is; setting `cutoff=...` masks edges past the value.

**Equivariant μ path is opt-in at the encoder level.** Setting
``Allegro(expose_tensor_track=True)`` does two things: writes the final
layer's tensor features to ``("edges","edge_tensor_features")`` AND flips
``_build_layer_irreps``'s last layer from "scalars-only" to "full SH irreps
stack" so the l=1 (and l=2) blocks survive into that final write. Users who
only want energies pay nothing — `expose_tensor_track=False` is the default.

### Quadrupole representation

5-component traceless symmetric basis ``(N, 5)``, NOT ``(N, 3, 3)``: the
traceless symmetric ``3 × 3`` tensor has exactly 5 degrees of freedom; storing
it as ``(N, 3, 3)`` carries 4 redundant numbers per atom and forces a reshape
when the v1 ``cuequivariance`` ``l=2`` projection lands.

## Invariants

| Property | Holds in v0? | Notes |
|----------|--------------|-------|
| Translation invariance of ``E_es`` | ✅ | depends on `bond_dist`, not `pos` |
| Rotation invariance of ``E_qq`` | ✅ | depends only on `|r_ij|` and scalar `q` |
| Rotation invariance of charge prediction | ✅ | `q_head` consumes scalars only |
| Permutation equivariance of moments | ✅ | by construction (edge / atom indexing) |
| Per-graph charge conservation | ✅ | linear projection enforces ``Σ_i q_i = Q_tot,g`` |
| Smoothness at encoder cutoff for ``q`` | ✅ | inherited from `edge_features` (Allegro `u(r_ij)` gating) |
| Rotation equivariance of ``μ`` head | ✅ | inlined scalar-gated l=1 readout over Allegro tensor track; tested in `test_multipole_symmetry.py` |
| Rotation equivariance of ``Θ`` head | ✅ | inlined scalar-gated l=2 readout over Allegro tensor track (`l_max >= 2`); transforms under Wigner D⁽²⁾, tested in `test_multipole_symmetry.py` |
| Energy conservation under autograd forces | ✅ | ``E_qq`` differentiable in `pos` via `bond_dist`; QM9 doesn't exercise it (no force labels) |

**Units.** Predicted ``q`` is dimensionless (atomic units of charge `e`).
``pos`` is Å. ``coulomb_constant = 14.399645 eV·Å·e⁻²`` so ``E_qq`` lands in
**eV**, matching the AtomicDress→eV training convention. Molecular dipole
``μ_mol`` is in ``e·Å``; QM9's `mu` is in **Debye** (1 D ≈ 0.20819 e·Å), so
the training script converts before computing the auxiliary loss.

**Dtype.** All trainable layers use `molix.config.ftype` (consistent with
`Allegro`, `EdgeEnergyHead`).

## Reference

References for what this layer **actually implements**:

* **Equivariant μ readout & molecular-dipole loss** —
  Schütt, Unke, Gastegger, *PaiNN: Equivariant Message Passing for the
  Prediction of Tensorial Properties and Molecular Spectra*, ICML 2021,
  arXiv:2102.03150. The `μ_mol = Σ_i q_i r_i + Σ_i μ_i` decomposition
  (PaiNN's `DipoleMoment` head) and the per-channel scalar-gated l=1
  message structure (`v_j ← φ(s_j) ⊙ v_j`) are PaiNN's contribution.
* **Pair-centred edge encoder** — Musaelian et al., *Allegro*,
  *Nature Communications* **14**, 579 (2023). Source of the
  ``("edges","edge_features")`` representation and the tensor track that
  ``PermMultipoleHead``'s μ / Θ readouts consume via
  ``expose_tensor_track=True``.
* **Multipole interaction kernels** — Stone, *The Theory of Intermolecular
  Forces*, 2nd ed. (2013), §3. Source of the higher-order energy term
  formulae through ``l = 1`` (``qq`` / ``qm`` / ``mm``) implemented;
  ``qt`` / ``mt`` / ``tt`` tracked as TODO.

Cited but **not** implemented here (would be a separate `QEqLayer`):

* Fuchs, Sanocki, Zavadlav, *CELLI: Charge Equilibration Layer for
  Long-range Interactions*, npj Comput. Mater. **11**, 71 (2025),
  https://doi.org/10.1038/s41524-025-01790-4. CELLI obtains charges from a
  Qeq KKT solve (Lagrange multiplier on `1ᵀQ = Q_tot`) and supervises on
  energy + forces + per-atom Hirshfeld charges. **No** atomic-dipole head
  and **no** molecular-dipole loss. Different physics, different code path
  — see "Compatibility note" in the module docstring.

## Open questions

1. **Equivariant Θ head — done.** Both `μ` (l=1) and `Θ` (l=2) readouts
   are now inlined inside :class:`PermMultipoleHead`, sharing one private
   ``_equivariant_moment_readout`` helper. They opt in via
   `Allegro(expose_tensor_track=True, l_max>=2)` which writes
   `("edges","edge_tensor_features")` AND flips
   `_build_layer_irreps(..., last_layer_keep_tensors=True)` so the final
   TP keeps the full SH irreps stack instead of being pruned to scalars.
   QM9 has no quadrupole label, so the `Θ` validation is currently
   equivariance-only — supervision waits on a labelled dataset
   (SPICE, ANI-1x).
2. **Quadrupole basis convention.** ``(N, 5)`` is decided, but the exact
   ordering of the 5 traceless symmetric components (real spherical harmonic
   `l=2` order vs. Cartesian-derived Stone basis) needs to match whatever
   convention the v1 supervision targets and the higher-order kernels use.
3. **Coulomb cutoff vs encoder cutoff.** v0 reuses the encoder's neighbour
   list. For organic molecules at ``r_cut ≥ 5 Å`` this misses long-range tails
   even on QM9-sized systems. Open: should v1 ship a separate, longer-cutoff
   `MultipoleNeighborList` task, or rely on the encoder's growth alone?
4. **Charge projection vs. charge loss.** Projection (mean-residual
   subtraction) ensures ``Σ q = Q_tot`` exactly but its gradient w.r.t.
   `q_raw` lives entirely in the null space of the constraint —
   ``∂q_proj_i/∂q_raw_k = δ_{ik} − 1/N_g`` per graph, so
   ``Σ_k ∂L/∂q_raw_k = 0`` for *any* downstream loss. The projection
   therefore does NOT, on its own, train the head to predict near-neutral
   sums. Observed empirically: with `dipole=True` enabled, `q_head`'s
   bias drifts under Adam noise (see chat thread 2026-04-26),
   ``charge_sum_pre_proj`` grows monotonically while
   ``charge_sum_post_proj`` stays at floating-point noise. The fix is a
   soft auxiliary loss ``λ · MSE(Σ q_raw, Q_tot)`` in the training
   script; track separately rather than bake into the layer because the
   weight depends on the dataset's atomic-charge scale.
5. **Open Q1 of the chat thread (vector dipole supervision).** The QM9
   `mu` target is a magnitude (Debye); the script computes
   ``(‖μ_mol‖ − μ_tgt).abs().mean()``. With the now-equivariant `μ` head
   we *could* train against the vector ``μ_mol`` directly, but QM9's
   stored `mu` is a scalar — vector targets need either a different
   dataset (SPICE, ANI-1x have ESP-derived vector dipoles) or a re-run
   of the QM9 SDFs through a QC code to recover the dipole vector.
