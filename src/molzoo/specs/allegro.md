# Allegro — Specification

**Module:** `molzoo.allegro`
**Entry point:** `Allegro` (config: `AllegroSpec`)
**Paper:** Musaelian et al., *"Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics"*, **Nature Communications 14, 579 (2023)**.
arXiv: https://arxiv.org/abs/2204.05249 · Reference impl: https://github.com/mir-group/allegro

---

## 1. Scope & Purpose

Allegro is a **strictly local, pair-centered equivariant encoder**. Unlike message-passing GNNs (MACE, NequIP), Allegro never propagates information across atoms — every pair `(i, j)` with `r_ij ≤ r_cut` accumulates a local many-body representation computed only from the atoms in the `r_cut`-ball of `i`. This makes the receptive field exactly `r_cut` regardless of depth, which is critical for large-scale MD: atomic decompositions have no data dependencies beyond the cutoff, enabling spatial parallelism without halo-size growth.

In this repository, `molzoo.allegro` implements the **encoder only**. It emits per-edge, per-layer scalar features

```
edge_features : (n_edges, num_layers, num_scalar_features)
```

Downstream readout (scalar → pair energy), energy aggregation, and autograd-based forces live in `molpot` (see `PotentialComposer`, `LayerPooling`).

---

## 2. Input / Output Contract

### Inputs (`GraphBatch` TensorDict)

| Key                       | Shape            | Description                                                    |
|---------------------------|------------------|----------------------------------------------------------------|
| `("atoms", "Z")`          | `(N,)` int       | Atomic numbers                                                 |
| `("edges", "edge_index")` | `(E, 2)` int     | `[:, 0]` = source `i` (centre), `[:, 1]` = target `j` (neighbour) |
| `("edges", "bond_diff")`  | `(E, 3)` float   | `pos[j] − pos[i]` (source → target)                            |
| `("edges", "bond_dist")`  | `(E,)` float     | `‖bond_diff‖`                                                  |

The edge set is the full bidirectional neighbour list returned by `NeighborList(symmetry=True)`.

### Outputs

| Key                          | Shape                                 | Description                              |
|------------------------------|---------------------------------------|------------------------------------------|
| `("edges", "edge_features")` | `(E, L, num_scalar_features)`         | Per-layer scalar features on each edge   |

where `L = num_layers`.

---

## 3. Notation

| Symbol            | Meaning                                                                    |
|-------------------|----------------------------------------------------------------------------|
| `N`, `E`          | Number of atoms, edges                                                     |
| `L`               | Number of Allegro layers (`num_layers`)                                    |
| `u`               | Tensor channel multiplicity (`num_tensor_features`)                        |
| `F`               | Scalar channel multiplicity (`num_scalar_features`)                        |
| `ℓ_max`           | Maximum angular order (`l_max`)                                            |
| `r_cut`           | Radial cutoff (`r_max`)                                                    |
| `\mathcal{N}(i)`  | `{ j : (i,j) ∈ edges }` — neighbours of atom `i`                           |
| `Y^ℓ_m(r̂)`       | Real spherical harmonics, `‖Y(r̂)‖ = 1`                                    |
| `ir_mul` layout   | Flat layout with m-axis major and channel axis fast: `(sh_dim × u)`        |

The full equivariant irrep set at multiplicity `u` is
`irreps_u = u × (0e ⊕ 1o ⊕ 2e ⊕ … ⊕ ℓ_max^p)` with parity `p = (−1)^ℓ`,
and the single-channel spherical-harmonic irreps
`irreps_sh = 0e ⊕ 1o ⊕ 2e ⊕ … ⊕ ℓ_max^p`.
Writing `d_sh = dim(irreps_sh) = \sum_{ℓ=0}^{ℓ_max}(2ℓ+1)`.

---

## 4. Architecture Overview

```
GraphBatch
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ PairEmbedding (two-body)                                      │
│   - BesselRBF(r_ij) · PolynomialCutoff(r_ij)   → edge_radial  │
│   - SphericalHarmonics(r̂_ij)                   → edge_angular │
│   - Embed(Z_i) ⊕ Embed(Z_j)                    → type_embed   │
│   - MLP([edge_radial, type_embed])             → x_0  (scalar)│
│   - V_0 = (Linear(x_0)) · edge_angular         → V_0  (tensor)│
└───────────────────────────────────────────────────────────────┘
    │  (x_0, V_0, edge_angular, edge_cutoff)
    ▼
┌───────────────────────────────────────────────────────────────┐
│ AllegroLayer × L                                              │
│   1. w_ij   = Linear(x_{ℓ-1})                  (edge scalar)  │
│   2. v_i    = aggregate_k∈N(i)  w_ik·u(r_ik)·Y(r̂_ik)          │
│   3. TP_ij  = V_{ℓ-1,ij}  ⊗_CG  v_i                            │
│              ↓  equivariant Linear                            │
│              V'_{ij}    (shape = irreps_u)                    │
│   4. I_ij   = L=0 scalars of V'_ij                            │
│   5. x_ℓ    = a·x_{ℓ-1} + b·MLP([x_{ℓ-1}, I_ij])              │
│   6. V_{ℓ}  = Linear(x_ℓ) · V'_ij       (per-channel scale)   │
│   7. x_ℓ   ← x_ℓ · u(r_ij)                                    │
└───────────────────────────────────────────────────────────────┘
    │  (x_L, V_L)
    ▼
stack([x_1, …, x_L]) → ("edges", "edge_features")
```

Per-layer outputs are **stacked after** the per-layer cutoff multiplication so that any decoder applied to `edge_features[:, ℓ]` inherits the smooth `r_cut` boundary.

---

## 5. Module Specifications

### 5.1 `PairEmbedding`

**Role.** Produce the initial scalar (`x_0`) and tensor (`V_0`) pair features from purely two-body information — distances, directions, and atomic types.

**Inputs:** `Z (N,)`, `bond_dist (E,)`, `bond_diff (E, 3)`, `edge_index (E, 2)`.

**Outputs:**

| Name              | Shape                       |
|-------------------|-----------------------------|
| `scalar_features` | `(E, F)`                    |
| `tensor_features` | `(E, d_sh · u)` in ir_mul   |
| `edge_angular`    | `(E, d_sh)` (single-channel)|
| `edge_cutoff`     | `(E,)`                      |

**Equations.**

Radial basis (Bessel) with polynomial cutoff envelope:

$$
B_n(r_{ij}) \;=\; \sqrt{\tfrac{2}{r_\mathrm{cut}}} \; \frac{\sin\!\left(n\pi\, r_{ij}/r_\mathrm{cut}\right)}{r_{ij}} , \quad n = 1, \ldots, N_\mathrm{bessel}
$$

$$
u(r) \;=\; 1 \;-\; \tfrac{(p+1)(p+2)}{2}\,(r/r_\mathrm{cut})^{p} \;+\; p(p+2)\,(r/r_\mathrm{cut})^{p+1} \;-\; \tfrac{p(p+1)}{2}\,(r/r_\mathrm{cut})^{p+2}
$$
for `r ≤ r_cut`, else `0`. Default `p = 6`.

The edge radial feature is $\tilde{B}_n(r) = B_n(r) \cdot u(r)$.

Angular basis:

$$
Y^\ell_m(\hat{r}_{ij}), \quad \hat{r}_{ij} = \frac{\mathbf{r}_j - \mathbf{r}_i}{\|\mathbf{r}_j - \mathbf{r}_i\| + \varepsilon}
$$

Type embedding preserves directionality by **concatenation** (not product):

$$
\mathbf{t}_{ij} \;=\; [\; \mathbf{e}(Z_i) \,;\; \mathbf{e}(Z_j) \;] \in \mathbb{R}^{2 d_\text{type}}
$$

Scalar MLP (default: two hidden layers of size `F`, SiLU activation, **no** activation after the final linear):

$$
\mathbf{x}_{0,ij} \;=\; \mathrm{MLP}\!\bigl(\bigl[\; \tilde{\mathbf{B}}(r_{ij}) \,;\, \mathbf{t}_{ij} \bigr]\bigr) \in \mathbb{R}^{F}
$$

Initial tensor features. A per-channel environment weight is linearly projected from the scalars, then broadcast against the single-channel spherical harmonics to produce a `u`-channel tensor feature in `ir_mul` layout:

$$
w^{(0),c}_{ij} \;=\; \bigl(W_{\text{env},0}\,\mathbf{x}_{0,ij}\bigr)_c, \qquad c=1,\ldots,u
$$
$$
V^{(0)}_{ij,(ℓ,m),c} \;=\; Y^ℓ_m(\hat{r}_{ij})\; w^{(0),c}_{ij}
$$

By construction `V^{(0)}` already carries the correct irreducible content: for each `ℓ ∈ {0,…,ℓ_max}` it is `u` copies of `Y^ℓ`.

### 5.2 `AllegroLayer`

**Role.** Refine `(x, V)` for each edge using an equivariant tensor product with a **neighbour-aggregated** spherical-harmonic vector. The centre atom `i` is the only one that scatters — the neighbourhood is summed **before** the TP, so no information beyond `r_cut` ever enters the feature.

**Inputs (from previous layer or from `PairEmbedding`):**

| Name            | Shape                         | Notes                                |
|-----------------|-------------------------------|--------------------------------------|
| `x_{ℓ-1}`       | `(E, F)`                      | scalar track                         |
| `V_{ℓ-1}`       | `(E, d_sh · u)` ir_mul        | tensor track                         |
| `edge_angular`  | `(E, d_sh)`                   | reused `Y(r̂)`                       |
| `edge_cutoff`   | `(E,)`                        | `u(r_ij)` envelope                   |
| `edge_index`    | `(E, 2)`                      | connectivity                         |
| `n_nodes`       | `int`                         | for scatter buffer                   |

#### Step 1 — Edge env weight (per-channel)

$$
\mathbf{w}_{ij} \;=\; W_\text{env}\,\mathbf{x}_{\ell-1,ij} \in \mathbb{R}^{u}
$$

A vector with **one weight per channel** per edge, matching the MIR Allegro reference. (A prior version of this code collapsed `w_ij` to a single scalar; see §7.1 for history.)

#### Step 2 — Neighbourhood aggregation

For each centre `i`, sum over its neighbours `k ∈ \mathcal{N}(i)` the envelope-weighted, per-channel-weighted spherical harmonics:

$$
\tilde{\mathbf{v}}_{i,c} \;=\; \sum_{k \in \mathcal{N}(i)} w_{ik,c}\, u(r_{ik})\; \mathbf{Y}(\hat{\mathbf{r}}_{ik}) \in \mathbb{R}^{d_{sh}}, \qquad c = 1,\ldots,u
$$

**Normalisation (Allegro SI).** If a dataset-wide constant is known:

$$
\mathbf{v}_i \;=\; \tilde{\mathbf{v}}_i \;\big/\; \sqrt{\overline{|\mathcal{N}|}}
$$

otherwise the per-node, cutoff-weighted fallback:

$$
\mathbf{v}_i \;=\; \tilde{\mathbf{v}}_i \;\big/\; \sqrt{\max\!\bigl(1,\; \sum_{k \in \mathcal{N}(i)} u(r_{ik})\bigr)}
$$

The cutoff weighting in the denominator is deliberate: an out-of-cutoff edge contributes `0` to both numerator and denominator, preserving the smooth `r → r_cut` limit.

For each edge `(i, j)` we then gather the centre's aggregate:

$$
\mathbf{v}_{ij,c} \;\equiv\; \mathbf{v}_{i,c}
$$

Note `\mathbf{v}_{ij}` is now a **`u`-channel** `(d_sh, u)` tensor in `ir_mul` layout — required for the per-channel TP below.

#### Step 3 — Equivariant tensor product (per-channel, ``u,iu,ju,ku+ijk``)

Per-channel tensor product: both operands share multiplicity `u`, each CG path has a learnable per-channel scalar weight `θ^{(ℓ_1,ℓ_2,ℓ_3),c}`:

$$
(\mathrm{TP})_{(ℓ_3,m_3),c} \;=\; \sum_{\substack{ℓ_1,m_1\\ℓ_2,m_2}} C^{(ℓ_1,m_1)(ℓ_2,m_2)}_{(ℓ_3,m_3)}\; \theta^{(ℓ_1,ℓ_2,ℓ_3),c}\; V_{(ℓ_1,m_1),c}\; v_{(ℓ_2,m_2),c}
$$

where `C` are Clebsch–Gordan coefficients. Filtered to `|ℓ_1 − ℓ_2| ≤ ℓ_3 ≤ ℓ_1 + ℓ_2` with compatible parity. This is the reference Allegro TP (subscripts `u,iu,ju,ku+ijk`), executed via `allegro_uuu_descriptor` + `cuet.SegmentedPolynomial`.

An equivariant linear then projects the TP output back into `irreps_u`:

$$
\mathbf{V}'_{ij} \;=\; W_\text{proj}\,\mathrm{TP}(\mathbf{V}_{\ell-1,ij},\, \mathbf{v}_i)
$$

#### Step 4 — Scalar invariants

Extract the `ℓ=0` block of `V'_ij` (the first `u` components in `ir_mul` layout):

$$
\mathbf{I}_{ij} \;=\; \mathbf{V}'_{ij}\bigl[\,:,\, 0\bigr] \in \mathbb{R}^{u}
$$

These are **rotationally invariant** by construction.

#### Step 5 — Residual scalar update (Allegro SI)

$$
\mathbf{x}_{\ell,ij} \;=\; a\,\mathbf{x}_{\ell-1,ij} \;+\; b\,\mathrm{MLP}\!\bigl([\,\mathbf{x}_{\ell-1,ij}\,;\, \mathbf{I}_{ij}\,]\bigr)
$$

with

$$
a \;=\; \frac{1}{\sqrt{1+\alpha^2}}, \qquad b \;=\; \frac{\alpha}{\sqrt{1+\alpha^2}}, \qquad a^2 + b^2 = 1
$$

Default `α = 0.5`. The unit-norm parameterisation keeps forward activations and gradients at unit scale across depth.

The latent MLP is `Linear → SiLU → … → Linear` (final layer has no activation) with output dimension `F`.

#### Step 6 — Tensor update (per-channel scaling)

$$
w^{(\ell),c}_{ij} \;=\; (W_\text{env,\ell}\,\mathbf{x}_{\ell,ij})_c
$$
$$
V_{\ell,ij,(ℓ',m'),c} \;=\; w^{(\ell),c}_{ij}\; V'_{ij,(ℓ',m'),c}
$$

Scalar × equivariant = equivariant, so `V_ℓ` remains a valid `irreps_u` tensor.

#### Step 7 — Post-layer cutoff (applied by `Allegro`, not the layer)

$$
\mathbf{x}_{\ell,ij} \;\leftarrow\; u(r_{ij})\; \mathbf{x}_{\ell,ij}
$$

This happens in the top-level `Allegro.forward` between layers so that bias and type-embedding paths in subsequent env weights also inherit the smooth cutoff. Without it, deep stacks (≥3 layers) can leak non-zero activations past `r_cut`.

### 5.3 `Allegro` (top-level)

**Role.** Orchestrate the embedding + `L` layers, apply the post-layer cutoff, stack per-layer scalars, and write the result into the `GraphBatch`.

**Pseudocode.**

```
x, V, Y, u = PairEmbedding(Z, bond_dist, bond_diff, edge_index)
per_layer = []
for ℓ in 1..L:
    x, V = AllegroLayer[ℓ](x, V, Y, u, edge_index, N)
    x = x * u(r)                                      # post-layer cutoff
    per_layer.append(x)
td["edges", "edge_features"] = stack(per_layer, dim=1)  # (E, L, F)
```

---

## 6. Mathematical Properties

The encoder is invariant/equivariant with respect to the following group actions:

| Transformation                              | Property on `edge_features` |
|---------------------------------------------|-----------------------------|
| `r_i → r_i + t` (translation)               | **invariant**               |
| `r_i → R r_i` with `R ∈ O(3)`               | **invariant**¹              |
| Permutation of atoms (global relabelling)   | **equivariant** (edges follow their endpoints) |
| Time reversal / parity                      | **invariant**²              |

¹ The output is scalar (`ℓ=0`) per edge per layer; the tensor track `V` is O(3)-equivariant intermediately.
² Parity: only scalars are emitted; all odd-parity content is absorbed inside the TP path.

**Locality.** For any pair `(i, j)`, `edge_features[i,j]` depends only on `{k : r_ik ≤ r_cut ∨ r_jk ≤ r_cut}` via `Y(r̂_ik)` (layer 1). Because `v_i` at layer `ℓ+1` is still formed from `Y(r̂_ik)` only, the receptive field does **not** expand with depth — it stays `r_cut`. This is the defining property of Allegro.

**Smoothness at `r_cut`.** By §5.1–5.2, every term contains at least one factor of `u(r_ij)` or `u(r_ik)`. Since `u(r_cut) = 0` together with `u'(r_cut) = 0` for `p ≥ 2`, `edge_features` and their first derivative with respect to `r_ij` both vanish at the cutoff, giving `C^1`-continuous forces.

---

## 7. Implementation Notes & Deviations from the Paper

This section is the one to consult when reconciling numerical results with the reference implementation.

### 7.1 Per-channel env weight (matches reference)

The paper (and the MIR reference) use an edge-specific **per-channel** weight `w_ij ∈ ℝ^u` inside the neighbour sum:

$$
v_{i,c} = \sum_k w_{ik,c}\; u(r_{ik})\; Y(\hat{r}_{ik}) \in \mathbb{R}^{d_{sh} \times u}
$$

This produces a `u`-channel node aggregate that feeds a per-channel tensor product (`u,iu,ju,ku+ijk`, see `allegro_uuu_descriptor`). `AllegroLayer` wires this via `cuet.SegmentedPolynomial`; the fast `ChannelWiseTensorProduct` (which enforces single-channel RHS) is no longer used.

**History.** A prior revision collapsed the env weight to a single scalar per edge to keep the fast TP kernel. That simplification was lossy — roughly equivalent to the reference's `path_channel_coupling=False` with the extra restriction that all CG paths share the weight — and was observed to hurt QM9 reproduction. It has been removed in favour of the per-channel descriptor.

### 7.2 A failed factorisation — for the record

A prior version attempted to factor the paper formula as `TP(w_ij · V_ij, Σ_k Y_k)` via "bilinearity". **This is incorrect.** `w_ij` depends on the edge being updated `(i, j)`, while the paper's weights `w_ik` depend on the neighbour being summed `k`. The two sums are not bilinear in the same variables and cannot be swapped. The current formulation is an explicit approximation, not an algebraic rewrite.

### 7.3 Cutoff weighting of the neighbour count

When falling back to the per-node `1/√|N(i)|` normalisation (i.e., `avg_num_neighbors=None`), the denominator **must** be the cutoff-weighted sum `Σ u(r_ik)` and not the raw `|N(i)|`. Using the raw count silently counts an almost-out-of-range neighbour equally with a close one, breaking the smooth cutoff limit. The fix is checked by the symmetry / cutoff tests.

### 7.4 `ir_mul` layout

All equivariant tensors (`V_ℓ`, TP outputs) use `cuequivariance.ir_mul` — m-components major, channels fast. Under uniform multiplicity `u`, a flat tensor of size `d_sh · u` reshapes as `(d_sh, u)` with no extra work, which is why `_env_weight_harmonics` and `_scale_by_channel` are simple outer-product / elementwise-scale helpers rather than per-`ℓ` loops.

### 7.5 What is **not** in this implementation

- **Per-species scale/shift on the final pair energy.** Handled downstream by `molpot`.
- **Pair-energy readout MLP.** Belongs to the readout layer, not to the encoder.
- **Chemical embedding with learnable one-hot expansion.** Replaced by `nn.Embedding` with concatenation (direction-preserving) rather than multiplication.
- **Tensor-track exposure.** The encoder emits only scalar `edge_features`; the equivariant `V` track is internal. Downstream tasks requiring equivariant features would need a widened out-key.

---

## 8. Configuration (`AllegroSpec`)

| Field                  | Type           | Default | Constraint        | Description                                                   |
|------------------------|----------------|---------|-------------------|---------------------------------------------------------------|
| `num_elements`         | `int`          | —       | `> 0`             | Atomic-number embedding table size                            |
| `num_scalar_features`  | `int`          | `64`    | `> 0`             | `F` — scalar channel multiplicity                             |
| `num_tensor_features`  | `int`          | `16`    | `> 0`             | `u` — tensor channel multiplicity                             |
| `r_max`                | `float`        | —       | `> 0`             | `r_cut` in Å                                                  |
| `num_bessel`           | `int`          | `8`     | `> 0`             | Bessel basis functions                                        |
| `l_max`                | `int`          | `2`     | `≥ 0`             | `ℓ_max`                                                       |
| `num_layers`           | `int`          | `2`     | `> 0`             | Depth `L`                                                     |
| `poly_p`               | `int`          | `6`     | `≥ 1`             | Polynomial cutoff exponent                                    |
| `scalar_mlp_hiddens`   | `list[int]?`   | `None`  | —                 | Two-body MLP widths (default `[F, F]`)                        |
| `latent_mlp_hiddens`   | `list[int]?`   | `None`  | —                 | Per-layer latent MLP widths (default: single linear, no hidden)|
| `avg_num_neighbors`    | `float?`       | `None`  | —                 | If set, dataset-wide norm; else per-node fallback             |
| `residual_alpha`       | `float`        | `0.5`   | —                 | `α` in the residual latent update                             |

---

## 9. Complexity

| Operation per forward call     | Cost                                 |
|--------------------------------|--------------------------------------|
| `PairEmbedding`                | `O(E · (N_bessel + F²))`            |
| `AllegroLayer` × L             | `O(L · E · (F² + d_sh · u · K_CG))` |
| Scatter aggregation            | `O(E · d_sh)` per layer              |

`K_CG` is the number of admissible `(ℓ_1, ℓ_2, ℓ_3)` paths at `ℓ_max`. Memory is dominated by the tensor track `V ∈ ℝ^{E × d_sh × u}`.

---

## 10. Validation

The following are checked in `tests/test_molzoo/`:

| Test file                         | Property                                                             |
|-----------------------------------|----------------------------------------------------------------------|
| `test_allegro.py`                 | Forward shape, scalar output type, determinism, gradient flow        |
| `test_allegro_parity.py`          | Parity / reflection behaviour of the scalar output                   |
| `test_symmetry.py`                | Rotation invariance, translation invariance, permutation equivariance|

Any change to §5 (module equations) that alters numerical behaviour must update these tests, and vice versa.

---

## 11. Experiment Log

Per-run results (MAE, forward / backward timings, commit, dataset, config tag)
live in the sibling file `allegro_experiments.csv`. Append rows with
`/molzoo-spec-log allegro`; retrieve spec + drift context with
`/molzoo-spec-lookup allegro <topic>`. Investigation notes produced in response
to flagged runs live under "Run-linked investigations" in
`allegro_walkthrough.md`, with each heading slug referenced back from the CSV
row's `note_ref` column.

---

## 12. Changelog Anchors

When the implementation diverges from this spec, update **both** the relevant §5 equation and the deviation in §7. Do not leave the spec stale — a silently wrong spec is worse than no spec at all.
