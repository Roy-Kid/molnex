# Allegro — Code / Spec / Reference Walkthrough

Three-way audit of:

| Source      | Path                                                     |
|-------------|----------------------------------------------------------|
| **Code**    | `molnex/src/molzoo/allegro.py`                           |
| **Spec**    | `molnex/src/molzoo/specs/allegro.md`                     |
| **Paper**   | Musaelian et al., *Nat. Commun.* 14, 579 (2023), arXiv:2204.05249 |
| **Ref impl**| `mir-group/allegro@main` (GitHub, 2026 snapshot)          |

Legend for the verdicts in each section:

- ✅ **consistent** — code, spec, and the authoritative source (paper) agree.
- ℹ️ **intentional deviation** — code deviates from the paper or reference, and the spec documents it.
- ⚠️ **spec mismatch** — spec does not correctly describe what the code actually does. **Action required.**
- 🆚 **reference drift** — reference implementation has drifted from the paper; we follow the paper.

---

## 0. Executive Summary

| # | Topic                                    | Verdict | Action                                   |
|---|------------------------------------------|---------|------------------------------------------|
| 1 | Bessel RBF formula                       | ⚠️     | Spec missing `+ε` and shift/scale norm   |
| 2 | PolynomialCutoff                         | ✅     | —                                        |
| 3 | Spherical harmonics                      | ✅     | —                                        |
| 4 | Type embedding (concat, not product)     | ℹ️     | OK — documented as a simplification      |
| 5 | Scalar MLP final activation              | ⚠️     | Code has SiLU after final linear; spec says "no activation after final" |
| 6 | Initial tensor V₀ per-channel weighting  | ℹ️     | OK — one scalar per channel (vs. paper per-channel-per-irrep) |
| 7 | Edge env weight `w_ij` in layer          | ℹ️     | OK — single scalar vs. paper per-channel (documented §7.1) |
| 8 | Neighbour aggregation formula            | ✅     | —                                        |
| 9 | Cutoff re-multiplication inside layer    | ℹ️/🆚  | OK — stricter than ref; improves stability |
| 10| Tensor product variant                   | ℹ️     | OK — `ChannelWiseTP` (right single-channel) vs. ref `uuu` |
| 11| Scalar invariants extraction             | ✅     | —                                        |
| 12| Residual scalar update (α-residual)      | ✅/🆚  | We follow paper; reference dropped it    |
| 13| Tensor track per-channel re-scale        | ℹ️     | Our addition; not in reference           |
| 14| Post-layer cutoff on scalar track        | ℹ️/🆚  | Our addition; not in reference. Spec documents why. |
| 15| Output shape                             | ℹ️     | `(E, L, F)` stack (ours) vs. `(E, F(L+1))` concat (ref) — our spec states this clearly |

Headline: **code matches spec and paper for all physics**; there are two spec-text bugs to fix (#1, #5) and one documentation gap about the reference having drifted from the paper (#12).

---

## 1. PairEmbedding — Bessel RBF

**Paper (SI eq. S3):**
$$
B_n(r) = \sqrt{\tfrac{2}{r_{cut}}} \cdot \frac{\sin(n\pi r / r_{cut})}{r}, \quad n = 1, \ldots, N_{bessel}
$$

**Reference impl.** `BesselEdgeLengthEncoding` — raw Bessel, no shift/scale. Trainable flag default `False`.

**Our code** (`molrep/embedding/radial.py:117–135`):
```python
def _raw_basis(self, r):
    return prefactor * sin(rr * freqs) / (rr + eps)
# then, when normalize=True (default):
def forward(self, r):
    phi = self._raw_basis(r)
    return (phi - mu) / sigma      # μ, σ estimated on r ~ U(0, r_cut)
```
So the effective formula is
$$
B_n(r) = \frac{1}{\sigma_n} \left( \sqrt{\tfrac{2}{r_{cut}}} \cdot \frac{\sin(n\pi r / r_{cut})}{r + \varepsilon} - \mu_n \right).
$$

**Our spec (§5.1)** states the *raw* form and omits both the `+ε` and the shift/scale. Functionally this is a per-channel centered-normalised Bessel basis as recommended in the Allegro SI ("normalize the Bessel basis so the MLP receives inputs of roughly unit variance"), so the physics is right — but the spec text is incomplete.

**Verdict:** ⚠️ spec mismatch. **Action:** update §5.1 to show the normalised form and a footnote about `normalize=True` being the default.

---

## 2. PolynomialCutoff

**Paper / ref / ours (all identical):**
$$
u(r) = 1 - \tfrac{(p+1)(p+2)}{2}x^p + p(p+2)\,x^{p+1} - \tfrac{p(p+1)}{2}\,x^{p+2}, \quad x = r/r_{cut}, \ r < r_{cut}
$$
$$
u(r) = 0 \quad \text{for } r \ge r_{cut}, \qquad p = 6 \text{ (default)}.
$$

Reference in `molrep/embedding/cutoff.py:94–162`. Spec §5.1 has the exact same expression.

**Verdict:** ✅ consistent across all sources.

---

## 3. Spherical harmonics

**Paper / ref / ours:** real `SH_ℓ^m(r̂)` with `ℓ = 0…ℓ_max`, unit-norm input. Dim per edge = `(ℓ_max+1)² ≡ d_sh`.

`molrep/embedding/angular.py:48–104` wraps `cuequivariance_torch.SphericalHarmonics` with `normalize=True`. Reference uses `edge_sh_normalize=True, normalization="component"`.

**Verdict:** ✅ consistent.

---

## 4. Type embedding — how Z_i and Z_j combine

**Paper:** one-hot(Z_i) ⊗ one-hot(Z_j) (pair-type outer product) → linear.

**Reference impl.** `ProductTypeEmbedding`:
1. Two `nn.Embedding`s (center, neighbor), each dim `d_type/2`.
2. Concatenate along channel → `d_type`.
3. Project Bessel features through a linear to `d_type`.
4. **Element-wise product** of the concat embedding and the projected Bessel.
5. Feed to `ScalarMLP`.

**Our code** (`allegro.py:251–313`):
1. Two `nn.Embedding`s (one shared parameter table, looked up twice).
2. Concatenate along channel → `2·d_type`.
3. **Concatenate** with `edge_radial`.
4. Feed to `scalar_mlp`.

**Our spec (§5.1)** describes concat + MLP, which matches the code.

This is a *mild* simplification of the reference:
- The reference's product gates the radial channels by the type pair.
- Ours lets the first linear of the MLP learn that gating if it wants to.

Both are O(3)-symmetric and permutation-equivariant; the expressivity difference is empirical.

**Verdict:** ℹ️ intentional deviation, documented in code comments and in spec §5.1. Should be called out in the spec's §7 "Deviations" section too, but that is cosmetic.

---

## 5. Scalar MLP activation pattern

**Reference impl.** `ScalarMLP`: `[Linear → SiLU] × (depth-1) → Linear` — **no activation after the final linear.**

**Our code** (`allegro.py:259–265`):
```python
for h in hiddens:
    layers.append(nn.Linear(prev, h))
    layers.append(nn.SiLU())      # <-- applied after EVERY linear, incl. final
```
SiLU **is** applied after the final linear.

**Our spec (§5.1)** says the MLP has "*no activation after the final linear*" — which is **false** for the current code.

**Verdict:** ⚠️ spec mismatch.

**Action:** one of
- (a) Remove the trailing `nn.SiLU()` on the last iteration in the code to match the reference / spec. Cleaner and gives the tensor track `V_0` unbounded per-channel weights (current `SiLU` squashes env weights toward (0, ∞)).
- (b) Fix the spec to say "SiLU is applied after every linear, including the final one" and add a note about the one-sided squash.

Option (a) is the more standard convention and what the reference does — this should be the fix unless the user has a reason to keep the current behaviour.

The identical pattern is NOT present in the **latent** MLP (`allegro.py:441–447`), which correctly omits the final activation. So the bug, if it is a bug, is localised to the two-body scalar MLP only.

---

## 6. Initial tensor features V₀

**Code** (`allegro.py:316–319`):
```python
env_weights = self.tensor_env(scalar_features)         # (E, u)
tensor_features = _env_weight_harmonics(edge_angular, env_weights, u)
# V_0[e, ℓm, c] = Y_{ℓ,m}(r̂_{ij}) * env_weights[e, c]
```
One scalar weight per channel `c` — applied uniformly across all `ℓ, m`.

**Reference impl.** `TwoBodySphericalHarmonicTensorEmbed` produces one scalar per `(channel, irrep)` via `MakeWeightedChannels`, i.e. `u × num_irreps` weights per edge, applied as `V_0[e, ℓm, c] = Y_{ℓm} * w[e, c, ℓ]` (same weight within an irrep block, different across irreps and channels).

**Our spec (§5.1)** matches our code (one scalar per channel, uniform across `ℓ`).

**Verdict:** ℹ️ intentional deviation. Strictly a subset of the reference: if our `tensor_env` produced an `(E, u, num_irreps)` tensor this would become the reference. A one-line widening if ever needed.

---

## 7. Layer env weight `w_ij`

**Paper (SI fig. S2 + eqs.):** `w_ij ∈ ℝ^u` per edge, per channel.

**Reference impl.** `env_weight_numel = u × num_irreps` with `weight_individual_irreps=True` (default) — even richer than the paper.

**Our code** (`allegro.py:434` and `allegro.py:500`):
```python
self.env_embed = nn.Linear(num_scalar_features, 1)
...
env_w = self.env_embed(scalar_features).squeeze(-1)    # (E,) — SINGLE scalar
```

**Our spec (§5.2 Step 1 and §7.1)** explicitly documents this as a simplification to keep the fast `ChannelWise` TP kernel path.

**Verdict:** ℹ️ intentional deviation, correctly documented. The rationale in §7.1 is accurate: adopting the reference's per-channel weight would require either the slow `uuu` kernel or a custom `cuEquivariance` descriptor (the `allegro_uuu_descriptor` function in `allegro.py` is already set up for that eventual port).

---

## 8. Neighbourhood aggregation

**Code** (`allegro.py:500–527`):
```python
weight = (env_w * edge_cutoff).unsqueeze(-1)      # (E, 1)
weighted_Y = edge_angular * weight                # (E, sh_dim)
node_Y.scatter_add_(0, src.expand_as(weighted_Y), weighted_Y)
# node_Y[i] = Σ_{k ∈ N(i)}  w_ik · u(r_ik) · Y(r̂_ik)

if avg_num_neighbors is not None:
    node_Y /= sqrt(avg_num_neighbors)
else:
    src_count = scatter_add(u(r_ik), src)          # Σ u(r_ik) per centre
    node_Y /= sqrt(max(1, src_count))
```

**Spec (§5.2 Step 2):** identical math, with the same fallback-normalisation formula.

**Verdict:** ✅ code ↔ spec consistent.

**Reference drift:** the reference uses `AvgNumNeighborsNorm` unconditionally (requires dataset-wide statistics) and does **not** include `u(r_ik)` inside the sum or the denominator. We include both. Including `u(r_ik)` in the numerator is physically motivated (preserves smoothness at `r_cut`); including it in the denominator is necessary to keep the normalisation consistent with the numerator — see §9 below.

---

## 9. Cutoff inside the layer

**Paper (SI §2.2):** "The envelope is applied once, to the Bessel basis." Silent on whether it propagates.

**Reference impl.** Not applied inside layers. Edges past `r_cut` are excluded from the graph entirely.

**Our code:** applied inside the aggregation (`weighted_Y` multiplies by `u(r_ij)`) **and** post-layer (`x_ℓ ← x_ℓ · u(r_ij)` between layers in `Allegro.forward:699–712`).

**Our spec (§5.2 Step 2 and Step 7):** documents both applications and justifies them.

**Verdict:** ℹ️ intentional deviation / 🆚 deliberate departure from reference.

**Rationale (already captured in spec):** without post-layer cutoff, type-embedding and bias paths in deeper layers leak non-zero activations past `r_cut`, which in turn causes a spike in the norm of `edge_features` for edges close to the boundary and destabilises forces. This became a problem on stacks of ≥3 layers in internal testing.

---

## 10. Tensor product variant

**Paper:** `TP(V_ij, v_i)` where both operands are `u`-channel, output is `u`-channel. Per-channel, per-path weights.

**Reference impl.** `Contracter` with subscripts `u,iu,ju,ku+ijk` or `u,pijk→uijk` (strided `uuu` with per-channel path weights).

**Our code** (`allegro.py:411–418`):
```python
self.tp = cuet.ChannelWiseTensorProduct(
    cue_irreps_in,         # u × (0e + 1o + 2e + ...)
    cue_irreps_sh,         # 1 × (0e + 1o + 2e + ...)   ← SINGLE channel
    layout=cue.ir_mul,
    shared_weights=True,
    internal_weights=True,
)
```

`ChannelWiseTensorProduct` is the cuEquivariance `uv,iu,jv,kuv+ijk` descriptor. With `v=1`, it reduces to `uu×1→u` — identical output multiplicity as reference, but the right operand is single-channel so the per-channel mixing across the neighbour-aggregated `v_i` is lost.

**Our spec (§5.2 Step 3 and §7.1):** correctly describes `ChannelWiseTensorProduct` with single-channel right operand.

**Verdict:** ℹ️ intentional deviation. Coupled to #7 — if we widen `w_ij` to per-channel, we also need to widen `v_i` to `u` channels, and then need a `uuu` kernel. All three changes travel together.

The module also carries an unused-at-runtime `allegro_uuu_descriptor` function for building exactly that kernel; hooking it up is a one-module change and would be the natural way to tighten our impl toward the paper.

---

## 11. Scalar invariants extraction

**Code** (`allegro.py:537`):
```python
invariants = new_tensor[:, : self.num_tensor_features]   # first u elements
```
In `ir_mul` layout with irreps `u × (0e ⊕ 1o ⊕ 2e ⊕ …)`, the first `u` flat entries *are* the `u` multiplicities of the `ℓ=0` block. These are rotationally invariant by construction.

**Spec (§5.2 Step 4):** describes the same extraction.

**Verdict:** ✅ consistent.

---

## 12. Residual scalar update — the α-residual

**Paper (SI):**
$$
\mathbf{x}_\ell = a\,\mathbf{x}_{\ell-1} + b\,\mathrm{MLP}([\mathbf{x}_{\ell-1},\,\mathbf{I}_\ell]), \quad a = \tfrac{1}{\sqrt{1+\alpha^2}},\ b = \tfrac{\alpha}{\sqrt{1+\alpha^2}}, \ \alpha = 0.5
$$

**Reference impl. (CURRENT, main branch 2026):** α-residual **was removed**. The MLP input now concatenates *all* previous per-layer scalars (DenseNet-style); there is no `alpha` parameter anywhere in the repo. The latent width grows with depth.

**Our code** (`allegro.py:452–456, 540`):
```python
alpha = float(residual_alpha)
denom = sqrt(1.0 + alpha * alpha)
self.residual_a = 1.0 / denom
self.residual_b = alpha / denom
...
updated_scalars = residual_a * scalar_features + residual_b * mlp_out
```
Implements the paper formula exactly, with `residual_alpha = 0.5` default.

**Our spec (§5.2 Step 5):** matches paper and code.

**Verdict:** ✅ code ↔ spec ↔ paper — we are faithful to the published model. 🆚 We diverge from the current reference implementation.

**Action:** add a **one-line note** in spec §7 (Implementation Notes) clarifying that the reference repo dropped α-residual in favour of a DenseNet input but we kept α-residual because (a) it is what the paper reports benchmarks on, and (b) it bounds the MLP input width, which is friendlier to `torch.compile` and fixed-shape allocation. Without this note, a future reader comparing to `mir-group/allegro` will be confused.

---

## 13. Tensor-track per-channel re-scaling

**Our code** (`allegro.py:543–546`):
```python
env_weights_out = self.tensor_env(updated_scalars)      # (E, u)
updated_tensor = _scale_by_channel(new_tensor, env_weights_out, u)
# V_ℓ[e, ℓm, c] = new_tensor[e, ℓm, c] * env_weights_out[e, c]
```

This step *does not exist* in the reference, where the tensor track is simply **replaced** by the TP output at each layer.

**Reasoning.** It keeps the tensor track coupled to the updated scalars — in effect, the latent MLP feedback enters both tracks. Without it, tensor features only evolve through the TP chain, and the scalar update is a "read-only" side channel. Coupling both directions is a design call, not a paper statement.

**Verdict:** ℹ️ our addition. Captured in spec §5.2 Step 6 but **not** acknowledged in §7 as a deviation from the reference. Worth one line in §7.

---

## 14. Post-layer cutoff on scalar track (Allegro.forward)

**Our code** (`allegro.py:699–712`):
```python
u = edge_cutoff.unsqueeze(-1)
for layer in self.layers:
    scalar_features, tensor_features = layer(...)
    scalar_features = scalar_features * u      # <-- post-layer cutoff on scalars
    per_layer_scalars.append(scalar_features)
```

Not in the reference. Our spec §5.2 Step 7 and §7 explain why (bias/type-embedding paths in deeper env weights would otherwise leak past `r_cut`).

**Verdict:** ℹ️ code ↔ spec consistent, and the rationale is correct.

---

## 15. Output shape — per-layer stack vs. DenseNet concatenation

**Reference impl.** Output is `concat([x_0, x_1, …, x_L], dim=-1)` → shape `(E, F·(L+1))`. Downstream MLP readout sees all layers at once.

**Our code** (`allegro.py:714`):
```python
td["edges", "edge_features"] = torch.stack(per_layer_scalars, dim=1)
# shape = (E, L, F), excludes layer-0 scalars
```

**Our spec (§2 and §5.3):** states `(E, L, F)` stack explicitly.

**Verdict:** ℹ️ intentional — our downstream pipeline uses `molpot.LayerPooling` to reduce the `L` axis with a learned or fixed per-layer weighting, which is more flexible than forcing the readout MLP to learn layer weights implicitly. The layer-0 two-body scalars are dropped on purpose because a pure two-body feature is already representable by the radial basis fed to the readout MLP if anyone needs it.

Worth a one-line mention in spec §7.

---

## 16. Symmetry claims in spec §6

| Claim                                      | Verified by                                   |
|--------------------------------------------|-----------------------------------------------|
| Translation invariance                     | Code: only uses `bond_dist`, `bond_diff` — no `pos`   ✅ |
| O(3) invariance of scalar output           | `invariants = new_tensor[:, :u]` is ℓ=0 block   ✅ |
| Permutation equivariance on atoms → edges  | Pure scatter + gather, no positional info   ✅ |
| Parity invariance of output                | Only ℓ=0 emitted, no pseudoscalars in current l_max range   ✅ (edge case: l_max odd would expose 0o; the extraction still returns a scalar but parity-odd if `l_max == odd` — nonstandard) |
| `C^1` smoothness at `r_cut`                 | `u(r_cut) = u'(r_cut) = 0` for `p ≥ 2` → every factor vanishes   ✅ |
| Locality (no receptive-field growth with depth) | `v_i` at every layer still built from `Y(r̂_ik)` with `k ∈ N(i)`, never from other atoms' aggregates   ✅ |

All six claims in §6 are correct.

---

## 17. Spec corrections — punch list

Required fixes to align the spec with the code:

1. **§5.1 Bessel RBF** — rewrite the formula to include `+ε` and the `(· − μ)/σ` normalisation, with a note that `normalize=True` is the default.
2. **§5.1 Scalar MLP** — either (a) change the code to drop the final SiLU, or (b) update the spec to reflect that SiLU is applied after *every* linear including the final. Recommend (a).
3. **§7 Deviations** — add one line each for:
   - Type-embedding combination (concat + MLP rather than product).
   - Tensor-track re-scaling (§5.2 Step 6) being our addition.
   - Post-layer cutoff on scalar track being our addition.
   - Output shape `(E, L, F)` stack vs. reference's concat.
   - α-residual kept (paper), reference dropped it.
4. **§9 Complexity** — optional: note that env-weight simplification (#7) reduces per-layer weight count from `O(u · num_irreps)` to `O(1)`.

Code changes that would tighten alignment *with the paper* (not required, but listed for planning):

- Widen `env_embed` from `Linear(F, 1)` to `Linear(F, u)` or `Linear(F, u · num_irreps)`.
- Aggregate `v_i` as a `u`-channel tensor (shape `(N, d_sh, u)`).
- Replace `ChannelWiseTensorProduct` with a `uuu`-kernel module built from `allegro_uuu_descriptor` (already defined in the module).

Those three changes travel together; none of them is safe in isolation because they are all coupled through the TP kernel.

---

## 18. Verdict

The current code is **internally consistent with the spec on all physics** except for (1) the Bessel normalisation footnote and (2) the final-layer activation in the scalar MLP. It implements the **paper** faithfully on the points where the reference has drifted (α-residual). The known simplifications (single scalar env weight, ChannelWise TP, type concat) are intentional, documented in the spec, and cleanly reversible if ever benchmarks demand the full paper expressivity.

The two documentation bugs (#1, #5) are cheap to fix and should be. The code change in #5 (drop final SiLU) is worth doing for paper-alignment and for consistency with the latent MLP.

---

## 19. Run-linked investigations

_Headings under this section are created by `molnex-scientist` when a run
logged in `allegro_experiments.csv` triggers an investigation (MAE regression,
dirty-tree run, or a user question that the rest of this walkthrough does not
cover). Each heading is `### run-<id>-<slug>` and is referenced back from the
CSV row's `note_ref` column. Entries are append-only and include the trigger,
paper citation, code location, and verdict._
