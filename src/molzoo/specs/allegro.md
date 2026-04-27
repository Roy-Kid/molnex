# Allegro — Specification

| Field | Value |
|-------|-------|
| Module | `molzoo.allegro` |
| Entry point | `Allegro` (config `AllegroSpec`) |
| Paper | Musaelian et al., *"Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics"*, Nature Communications 14, 579 (2023) |
| arXiv | https://arxiv.org/abs/2204.05249 |
| DOI | https://doi.org/10.1038/s41467-023-36329-y |
| Reference impl | `mir-group/allegro` (`allegro/nn/_allegro.py::Allegro_Module`) + `mir-group/nequip` (`nequip/nn/mlp.py`, `nequip/nn/embedding/_edge.py`) |
| Spec status | stable — verbatim port |

## 1. Scope & Boundary
- **Does:** Encode each edge `(i,j)` as a strictly-local invariant scalar feature stack. Output is the DenseNet concatenation of the two-body scalar embedding and every Allegro layer's freshly-produced scalar features, in the order `[x_0 ; x_1 ; … ; x_L]`. Receptive field is exactly `r_cut` regardless of depth.
- **Does NOT:** No readout MLP, no pair-energy aggregation, no force gradient, no per-species scale/shift, no exposure of the tensor track `V`. Those live in `molpot`.
- **Sits at:** Between `molix.data.NeighborList` (edge construction) and `molpot.heads.EdgeEnergyHead` / `molpot.PotentialComposer` (readout + energy/force).

### 1.1 I/O contract
| Direction | TensorDict path | Shape | Dtype | Source/sink |
|-----------|------------------|-------|-------|-------------|
| In | `("atoms","Z")` | `(N,)` | `int64` | `DataModule` |
| In | `("edges","edge_index")` | `(E,2)` | `int64` | `NeighborList` (`symmetry=True`); `[:,0]=src/center`, `[:,1]=dst/neighbor` |
| In | `("edges","bond_diff")` | `(E,3)` | float | `NeighborList` (`pos[dst]−pos[src]`) |
| In | `("edges","bond_dist")` | `(E,)` | float | `NeighborList` |
| Out | `("edges","edge_features")` | `(E, F·(L+1))` | float | downstream readout (DenseNet stack) |

`output_dim = num_scalar_features * (num_layers + 1)` is exposed as an attribute. Anything not listed MUST NOT be read or written by this module.

## 2. Paper ↔ Code Mapping
- One row per paper / reference concept; never silently absent.
- Status ∈ `matched` / `adapted` / `missing` / `unknown`. Non-`matched` rows MUST link to a §4 ID.

| Concept | Reference impl | This repo (file:line) | Status |
|---------|----------------|------------------------|--------|
| Bessel × cutoff edge length encoding `bessel = sinc(n·r/r_max)·n · u(r)` | `nequip/nn/embedding/_edge.py::BesselEdgeLengthEncoding` | `src/molzoo/allegro.py` (forward, bessel × cutoff block) | matched |
| Polynomial cutoff envelope `u(r)` (degree `p`) | `nequip/nn/embedding/cutoffs.py::PolynomialCutoff` | `src/molrep/embedding/cutoff.py::PolynomialCutoff` (used in `Allegro.__init__`) | matched |
| `ProductTypeEmbedding`: `type_embed × basis_linear(bessel)` (Hadamard) | `allegro/nn/_edgeembed.py::ProductTypeEmbedding` | `src/molzoo/allegro.py` (forward step 2) | matched |
| Type embedding via `cat(center_embed(Z_i), neighbor_embed(Z_j))` (split halves) | `allegro/nn/_edgeembed.py::ProductTypeEmbedding` | `src/molzoo/allegro.py` (`center_embed` / `neighbor_embed`) | matched |
| Two-body scalar embed = ProductTypeEmbedding output (no extra MLP) | `allegro/nn/scalarembed.py::TwoBodyBesselScalarEmbed` (just `bessel_encode + type_embed`) | `src/molzoo/allegro.py` (forward step 2 — directly fed to V_0 / first-layer projection, no `scalar_embed_mlp`) | matched |
| Single-channel spherical harmonics `Y(r̂)` | `nequip/nn/embedding/_edge.py::SphericalHarmonicEdgeAttrs` | `src/molrep/embedding/angular.py::SphericalHarmonics` (used in `Allegro.__init__`) | matched |
| `MakeWeightedChannels(SH, weights)` (per-irrep, per-channel weighting, `weight_individual_irreps=True`) | `allegro/nn/_strided/_channels.py::MakeWeightedChannels` | `src/molzoo/allegro.py::_make_weighted_channels` | matched |
| Initial tensor track `V_0 = MakeWeightedChannels(SH, env_embed_linear(twobody))` | `allegro/nn/tensorembed.py::TwoBodySphericalHarmonicTensorEmbed` | `src/molzoo/allegro.py` (`env_embed_linear` + `_make_weighted_channels` in forward) | matched |
| `env_embed_linear` input dim = `module_output_dim` (= `type_embed_dim`) | `allegro/nn/tensorembed.py::TwoBodySphericalHarmonicTensorEmbed` (`scalar_embedding_in_field` num_irreps) | `src/molzoo/allegro.py::Allegro.__init__` (`env_embed_linear`, `input_dim=type_embed_dim`) | matched |
| First-layer projection: `twobody → [scalar_features ; env_w_for_layer_0]` (single Linear) | `allegro/nn/_allegro.py::Allegro_Module.first_layer_env_embed_projection` | `src/molzoo/allegro.py::Allegro.__init__` (`first_layer_env_embed_projection`, `input_dim=type_embed_dim`) | matched |
| Per-edge env weights `env_w[ij]`, broadcast over single-channel SH → `env_w_edges` | `allegro/nn/_allegro.py::Allegro_Module` (forward, layer loop, `_env_weighter(tensor_basis, env_w)`) | `src/molzoo/allegro.py` (forward step 5a) | matched |
| Neighbour aggregation `env_w_scatter[i] = Σ_k env_w_edges[ik]` via `scatter_add_` | `allegro/nn/_allegro.py::Allegro_Module` (forward, `scatter(...)`) | `src/molzoo/allegro.py` (forward step 5b) | matched |
| Aggregate normalisation `1/√⟨\|N\|⟩` (dataset-wide constant; required) | `allegro/nn/_allegro.py::Allegro_Module.avg_num_neighbors_norm` | `src/molzoo/allegro.py` (`avg_num_neighbors_inv_sqrt`) | matched |
| Two-pass irreps construction (forward path-prune + backward dead-path prune) | `allegro/nn/_allegro.py::Allegro_Module.__init__:115-161` | `src/molzoo/allegro.py::_build_layer_irreps` | matched |
| Per-channel CG TP descriptor `u, iu, ju, ku + ijk` with **paths fused into shared per-ir output segments** (`(u, num_paths_to_ir)` weights) | `allegro/nn/_strided/_contract.py::Contracter` (`path_channel_coupling=True`, `irreps_out` constraint) | `src/molzoo/allegro.py::_allegro_uuu_descriptor` | matched |
| Per-layer TP `tp(V_{ℓ-1}, env_w_scatter[src])` directly produces `ir_out` (no extra equivariant linear) | `allegro/nn/_allegro.py::Allegro_Module.tps` (Contracter takes `irreps_out` directly) | `src/molzoo/allegro.py::Allegro.__init__` / forward (`self.tps`, no `tp_linears`) | matched |
| Last-layer pruning to scalars only (`0e × u`) | `allegro/nn/_allegro.py::Allegro_Module.__init__` (`ir_out = Irreps([(1, (0, 1))])`) | `src/molzoo/allegro.py::_build_layer_irreps` (`SCALAR_IRREPS` branch) | matched |
| Scalar invariants `I_ij = V_ℓ[ij][:u]` (L=0 block in ir_mul / first u channels in mul_ir) | `allegro/nn/_allegro.py::Allegro_Module` (forward; `tensor_features[:, :, :n_scalar]`) | `src/molzoo/allegro.py` (forward step 5d) | matched |
| DenseNet latent MLP input `cat([x_0, x_1, …, x_{ℓ-1}, I_ij])` | `allegro/nn/_allegro.py::Allegro_Module` (`accumulated_scalar_features + [scalars]`) | `src/molzoo/allegro.py` (forward step 5e) | matched |
| Latent MLP output split `[new_scalar ; env_w_next]` (env_w_next omitted on last layer) | `allegro/nn/_allegro.py::Allegro_Module` (`torch.narrow` of latents output) | `src/molzoo/allegro.py` (forward step 5f) | matched |
| `ScalarLinearLayer` variance-preserving init: `weight ~ U(-√3, √3)`, scaled by `α = gain/√fan_in`, `bias=False` | `nequip/nn/mlp.py::ScalarLinearLayer` | `src/molrep/embedding/scalar_mlp.py::ScalarLinearLayer` | matched |
| `ScalarMLPFunction`: `forward_weight_init=True`, gain `√2` for hidden layers, `1` for first layer | `nequip/nn/mlp.py::ScalarMLPFunction` | `src/molrep/embedding/scalar_mlp.py::ScalarMLPFunction` | matched |
| Final output: `cat([x_0, x_1, …, x_L])` (DenseNet stack, dim `F·(L+1)`) | `allegro/nn/_allegro.py::Allegro_Module` (return) | `src/molzoo/allegro.py` (forward step 6) | matched |
| Pair-energy readout MLP (linear, no nonlinearity, no bias) | `allegro/nn/edgewise.py::EdgewiseReduce` (+ paper Edge Energy MLP) | `src/molpot/heads/edge_energy.py::EdgeEnergyHead` (factored to `molpot`) | adapted (A1) |
| Edgewise reduce `/√⟨\|N\|⟩` and `/√2` for double-counted derivatives | `allegro/nn/edgewise.py::EdgewiseReduce` | `src/molpot/heads/edge_energy.py::EdgeEnergyHead` | matched (in `molpot`) |
| Per-species pair-energy scale/shift | reference `nequip` post-processing module | — (downstream `molpot`) | missing (A1) |
| Tensor-track output exposure (`forward_after_last_layer_tps`) | optional in reference | — (encoder is scalar-only) | missing (A2) |

## 3. Reference Implementation Alignment
- **Repo:** `mir-group/allegro` (Allegro layer + edge embeddings) and `mir-group/nequip` (MLP / radial / cutoff primitives). The encoder is a verbatim port; deviations are limited to what is enumerated in §4.

### 3.1 Identical
- **Layer architecture.** `Allegro_Module` does not use an α-residual; it accumulates per-layer scalar features in a DenseNet list and concatenates them at the end. `src/molzoo/allegro.py` mirrors this exactly: each layer appends a new `(E, F)` slice to `accumulated`, and the encoder returns `cat(accumulated)`.
- **Two-body scalar embed = ProductTypeEmbedding only.** Reference's `TwoBodyBesselScalarEmbed` is just `bessel_encode + ProductTypeEmbedding` — no extra MLP between bessel and the layer loop. The port matches: `twobody = type_embed × basis_linear(bessel)` of dim `type_embed_dim` is fed directly to `env_embed_linear` (V_0 weights) and `first_layer_env_embed_projection` (`twobody → [scalar_features ; env_w_0]`).
- **Env-weight wiring.** Reference computes the env weights for layer `ℓ+1` by slicing the *output* of layer `ℓ`'s latent MLP (one MLP, two output sub-blocks). The port reproduces that — `latents_out[:, :F]` is the new scalar feature, `latents_out[:, F:]` is `env_w` for the next layer (omitted on the last layer). There are no separate `env_embed` Linear layers.
- **Per-(irrep, channel) env weights.** Reference uses `MakeWeightedChannels(weight_individual_irreps=True)` to weight the single-channel SH basis by `(num_irreps × num_tensor_features)` scalars, producing `(E, irreps_dim · u)`. The port re-implements this in `_make_weighted_channels`.
- **Two-pass irreps build (forward + backward pruning).** Reference (`_allegro.py:115-161`) first prunes per-layer outputs to reachable irs (forward pass), then walks backward from the last (scalar) output and drops any earlier-layer arg ir that cannot eventually reach the final output. The port replicates both passes in `_build_layer_irreps`. For typical configurations (`l_max ≤ 3`, env spans all parities) the backward pass is a no-op at depth 0, but the port asserts this rather than silently dropping V_0 channels.
- **Per-channel CG TP with fused per-ir output segments.** Reference's `Contracter` with `path_channel_coupling=True` constrains the polynomial output to the caller-supplied `irreps_out` and uses weights of shape `(u, num_paths_to_ir)` per output ir. The port's `_allegro_uuu_descriptor` matches this by **fusing all CG paths producing the same `ir3` into one shared output segment** — every path adds an independent `(u,)` weight segment (operand 0) but routes its CG contribution into the segment for its `ir3`, so the trainable weight count per output ir is exactly `u · num_paths_to_ir`, identical to reference. There is **no** extra `cuet.Linear` between the TP and the next layer's TP input.
- **Last-layer pruning.** Reference prunes the last layer's output irreps to scalars only (no further TPs read from `V_L`). The port replicates this in `_build_layer_irreps` (the `layer_idx == num_layers - 1` branch).
- **Variance-preserving init.** Both `ScalarLinearLayer` and `ScalarMLPFunction` are direct ports of `nequip/nn/mlp.py` — uniform `(−√3, √3)` init, `α = gain/√fan_in` scaling at forward, `gain = √2` for hidden layers (relu/SiLU), `gain = 1` for the first layer (`forward_weight_init=True`), `bias = False` everywhere. Crucially, `EdgeEnergyHead` also uses `ScalarMLPFunction`, so the encoder's careful init is not undone by a default-init readout.
- **Cutoff smoothness.** The cutoff envelope `u(r)` is applied **once** to the bessel basis (`edge_radial = bessel * u(r)`). All downstream linear layers are bias-free, so any quantity that traces back to `edge_radial` is mapped to `0` at `r=r_cut`. There is no need for an additional encoder-level multiplicative gate.

### 3.2 Differs from reference
| Concern | Reference | This repo | Why | §4 ID |
|---------|-----------|-----------|-----|-------|
| Pair-energy readout / per-species scale-shift | bundled inside reference | factored out to `molpot.heads.EdgeEnergyHead` | encoder-only molzoo charter | A1 |
| Tensor-track output | exposed in some configurations (`forward_after_last_layer_tps`) | not exposed | encoder emits scalars only | A2 |

## 4. Adaptation Ledger
- One row per deviation — bundling forbids attribution.
- Risk: `low` (<1 % metric shift) / `medium` (1–5 %) / `high` (>5 % or symmetry break).
- Validation MUST resolve to a test path or a §7.4 `run_id`. `pending` only with a linked issue.

| ID | Change | Reason | Impact (metric, dataset) | Risk | Validation |
|----|--------|--------|---------------------------|------|------------|
| A1 | Pair-energy readout, per-species scale/shift, and energy aggregation factored out to `molpot.heads.EdgeEnergyHead`. The readout uses the same `ScalarMLPFunction` (variance-preserving init, no bias) so end-to-end behaviour matches the reference's bundled module. | molzoo charter: encoders emit features only. | n/a (architectural) | low | `tests/test_molzoo/test_allegro.py::TestEnergyInvariants`, `TestOverfitSingleBatch`. |
| A2 | Tensor track `V` is internal; encoder emits only scalar `edge_features`. | Downstream tasks needing equivariant features would widen the out-key contract; default is scalar-only. The reference has the same default; only the optional `forward_after_last_layer_tps` flag is omitted here. | n/a (scope) | low | §1.1 contract; `tests/test_molzoo/test_allegro.py::TestAllegroEncoder`. |

## 5. Mathematical Contract
- ONLY equations directly implemented and necessary for behavior.
- Each subsection: equation in display math + symbol→code table.
- Anything not implemented goes to §4 as `missing`, not here.

### 5.0 Notation

| Symbol | Meaning |
|--------|---------|
| `N`, `E` | Number of atoms, edges |
| `L` | Number of Allegro layers (`num_layers`) |
| `u` | Tensor channel multiplicity (`num_tensor_features`) |
| `F` | Scalar channel multiplicity (`num_scalar_features`) |
| `ℓ_max` | Maximum angular order (`l_max`) |
| `r_cut` | Radial cutoff (`r_max`) |
| `\mathcal{N}(i)` | `{ k : (i,k) ∈ edges }` — neighbours of atom `i` |
| `Y^ℓ_m(r̂)` | Real spherical harmonics, `component` normalisation |
| `ir_mul` layout | Per-irrep flat layout: `m` axis outer, channel axis inner-fast (`d_r · u` per irrep) |
| `d_sh` | `dim(irreps_sh) = Σ_{ℓ=0}^{ℓ_max}(2ℓ+1)` |
| `n_sh` | `len(irreps_sh) = ℓ_max + 1` |
| `W` | `_env_weight_numel = n_sh · u` (per-(irrep, channel) env weight count) |

The full equivariant irrep set at multiplicity `u` is `irreps_u = u × (0e ⊕ 1o ⊕ 2e ⊕ … ⊕ ℓ_max^p)` with parity `p = (−1)^ℓ`; the single-channel spherical-harmonic irreps `irreps_sh = 0e ⊕ 1o ⊕ 2e ⊕ … ⊕ ℓ_max^p`.

### 5.1 Edge length encoding (Bessel × polynomial cutoff)

$$
B_n(r_{ij}) \;=\; \mathrm{sinc}\!\left(\tfrac{n\, r_{ij}}{r_\mathrm{cut}}\right) \cdot n, \qquad n = 1, \ldots, N_\mathrm{bessel}
$$

$$
u(r) \;=\; 1 \;-\; \tfrac{(p+1)(p+2)}{2}\,(r/r_\mathrm{cut})^{p} \;+\; p(p+2)\,(r/r_\mathrm{cut})^{p+1} \;-\; \tfrac{p(p+1)}{2}\,(r/r_\mathrm{cut})^{p+2}
$$

for `r ≤ r_cut`, else `0`. `\mathrm{sinc}(z) = \sin(πz)/(πz)` (PyTorch convention). Default `p = 6`. The two factors are multiplied **once**:

$$
\tilde{B}_n(r) \;=\; B_n(r) \cdot u(r)
$$

This is the only place the cutoff is applied multiplicatively. The bias-free linear layers below preserve `0 → 0`, so the cutoff propagates implicitly through the whole pipeline.

| Symbol | Domain/shape | Implemented at |
|--------|--------------|----------------|
| `bessel_n` (buffer, `[1, n]`) | `(1, N_bessel)` | `src/molzoo/allegro.py:333–336` |
| `B(r)` | `(E, N_bessel)` | `src/molzoo/allegro.py:478–480` |
| `u(r)` | `(E,)` | `src/molzoo/allegro.py:337, 481` (`PolynomialCutoff`) |
| `\tilde{B}(r)` (`edge_radial`) | `(E, N_bessel)` | `src/molzoo/allegro.py:482` |

### 5.2 Two-body scalar embedding (`ProductTypeEmbedding` + `ScalarMLP`)

$$
\mathbf{t}_{ij} \;=\; \bigl[\; e_\text{c}(Z_i) \,;\, e_\text{n}(Z_j) \;\bigr] \in \mathbb{R}^{d_\text{type}}
$$

$$
\mathbf{s}_{ij} \;=\; \mathbf{t}_{ij} \;\odot\; W_\text{basis}\, \tilde{\mathbf{B}}(r_{ij}) \in \mathbb{R}^{d_\text{type}}
$$

$$
\mathbf{x}_{0,ij} \;=\; \mathrm{ScalarMLP}(\mathbf{s}_{ij}) \in \mathbb{R}^{F}
$$

`e_c`/`e_n` are independent `nn.Embedding` tables of width `d_type/2` (so `cat` produces `d_type`). `W_basis` is a single linear `(N_bessel → d_type, bias=False)`. `ScalarMLP` is the variance-preserving MLP from `molrep.embedding.scalar_mlp` with `bias=False`.

| Symbol | Domain/shape | Implemented at |
|--------|--------------|----------------|
| `e_c(Z)`, `e_n(Z)` | `(N_elements, d_type/2)` each | `src/molzoo/allegro.py:340–342` |
| `W_basis` | `(N_bessel, d_type)` | `src/molzoo/allegro.py:344–350` (`basis_linear`) |
| `t_ij` | `(E, d_type)` | `src/molzoo/allegro.py:485–487` |
| `s_ij` | `(E, d_type)` | `src/molzoo/allegro.py:488` |
| `x_0` | `(E, F)` | `src/molzoo/allegro.py:353–360, 491` (`scalar_embed_mlp`) |

### 5.3 Initial tensor track `V_0` (`TwoBodySphericalHarmonicTensorEmbed`)

Per-(irrep, channel) weights from a single Linear over `x_0`:

$$
\mathbf{w}^{(0)}_{ij} \;=\; W_\text{env,0}\,\mathbf{x}_{0,ij} \in \mathbb{R}^{W}, \quad W = n_\text{sh} \cdot u
$$

Weight the single-channel SH basis per (irrep, channel) using `MakeWeightedChannels`:

$$
V^{(0)}_{ij,(ℓ,m,c)} \;=\; Y^ℓ_m(\hat{r}_{ij}) \cdot w^{(0),(ℓ,c)}_{ij}, \qquad c=1,\ldots,u
$$

The output is laid out in `ir_mul` order: per irrep `r` of dim `d_r`, the `m` axis is outer and the channel axis is inner-fast, giving `d_r · u` consecutive scalars per irrep, total `d_sh · u`.

| Symbol | Domain/shape | Implemented at |
|--------|--------------|----------------|
| `Y(r̂)` | `(E, d_sh)` | `src/molzoo/allegro.py:363, 494–495` (`SphericalHarmonics`) |
| `W_env,0 x_0` | `(E, W)` | `src/molzoo/allegro.py:373–379, 496` (`env_embed_linear`) |
| `V_0` | `(E, d_sh · u)` (`ir_mul`) | `src/molzoo/allegro.py:497–502` (`_make_weighted_channels`) |

### 5.4 First-layer projection (one Linear emits scalar + env_w jointly)

$$
\bigl[\,\mathbf{f}_{0,ij}\,;\,\mathbf{w}^{(1)}_{ij}\,\bigr] \;=\; W_\text{first}\,\mathbf{x}_{0,ij}, \qquad
\mathbf{f}_{0,ij} \in \mathbb{R}^{F},\; \mathbf{w}^{(1)}_{ij} \in \mathbb{R}^{W}
$$

The accumulated scalar list is initialised with `f_0`:

$$
\mathrm{accumulated} \leftarrow \bigl[\, \mathbf{f}_{0,ij} \,\bigr]
$$

| Symbol | Domain/shape | Implemented at |
|--------|--------------|----------------|
| `W_first` | `(F, F + W)` | `src/molzoo/allegro.py:383–389` (`first_layer_env_embed_projection`) |
| `f_0`, `w^{(1)}` | `(E, F)`, `(E, W)` | `src/molzoo/allegro.py:504–508` |

### 5.5 Per-layer update `ℓ = 1, …, L`

Edgewise weighted SH using current env weights:

$$
\mathbf{E}_{ij}^{(ℓ)} \;=\; \mathrm{MakeWeightedChannels}\!\bigl(Y(\hat{r}_{ij}),\, \mathbf{w}^{(ℓ)}_{ij}\bigr) \in \mathbb{R}^{d_\text{sh} \cdot u}\;(\text{ir\_mul})
$$

Scatter to source nodes and apply dataset-wide aggregate norm:

$$
\bar{\mathbf{E}}_i^{(ℓ)} \;=\; \frac{1}{\sqrt{\overline{|\mathcal{N}|}}} \sum_{k \in \mathcal{N}(i)} \mathbf{E}_{ik}^{(ℓ)}
$$

Per-channel CG tensor product (subscripts `u, iu, ju, ku + ijk`) followed by an equivariant linear projection into the layer's pruned `ir_out`:

$$
V^{(ℓ)}_{ij} \;=\; W^{(ℓ)}_\text{proj}\;\mathrm{TP}\bigl(V^{(ℓ-1)}_{ij},\, \bar{\mathbf{E}}_{\,\mathrm{src}(ij)}^{(ℓ)}\bigr)
$$

Filtered to `|ℓ_1 − ℓ_2| ≤ ℓ_3 ≤ ℓ_1 + ℓ_2` with compatible parity. For `ℓ < L`, `ir_out = irreps_u`; for `ℓ = L`, `ir_out = u × 0e` (last-layer pruning). Scalar invariants are the L=0 block (the first `u` components in `ir_mul`):

$$
\mathbf{I}_{ij}^{(ℓ)} \;=\; V^{(ℓ)}_{ij}\bigl[\,:u\,\bigr] \in \mathbb{R}^{u}
$$

DenseNet latent MLP — input is the concatenation of all previously accumulated per-edge scalar features and the new invariants:

$$
\bigl[\,\mathbf{f}_{ℓ,ij}\,;\,\mathbf{w}^{(ℓ+1)}_{ij}\,\bigr] \;=\; \mathrm{MLP}^{(ℓ)}\!\Bigl(\bigl[\, \mathbf{f}_{0,ij}\,;\,\mathbf{f}_{1,ij}\,;\,\ldots\,;\,\mathbf{f}_{ℓ-1,ij}\,;\,\mathbf{I}_{ij}^{(ℓ)} \,\bigr]\Bigr)
$$

with output dim `F + W` for `ℓ < L` and `F` for `ℓ = L` (no env_w needed after the last layer). Append the new feature: `accumulated ← accumulated + [f_ℓ]`.

| Symbol | Domain/shape | Implemented at |
|--------|--------------|----------------|
| `E_ij^{(ℓ)}` | `(E, d_sh · u)` (`ir_mul`) | `src/molzoo/allegro.py:516–522` |
| `\bar{E}_i^{(ℓ)}` (scatter + norm) | `(N, d_sh · u)` | `src/molzoo/allegro.py:524–534` |
| `1/√⟨\|N\|⟩` | scalar (precomputed) | `src/molzoo/allegro.py:329` |
| TP descriptor (`u,iu,ju,ku+ijk`) | — | `src/molzoo/allegro.py:122–184` (`allegro_uuu_descriptor`) |
| `EquivariantPolynomialTP` | `(E, dim(tp.irreps_out))` | `src/molzoo/allegro.py:421–428, 537` |
| `cuet.Linear(W_proj)` | `(E, dim(ir_out))` | `src/molzoo/allegro.py:429–437, 538` |
| `I_ij^{(ℓ)}` | `(E, u)` | `src/molzoo/allegro.py:541` |
| `MLP^{(ℓ)}` | input `F·ℓ + u`, output `F + W` (or `F` if last) | `src/molzoo/allegro.py:444–461, 543–546` |
| `f_ℓ`, `w^{(ℓ+1)}` | `(E, F)`, `(E, W)` | `src/molzoo/allegro.py:549–552` |

### 5.6 Final output (DenseNet stack)

$$
\mathrm{edge\_features}_{ij} \;=\; \bigl[\, \mathbf{f}_{0,ij}\,;\, \mathbf{f}_{1,ij}\,;\, \ldots\,;\, \mathbf{f}_{L,ij} \,\bigr] \in \mathbb{R}^{F\cdot(L+1)}
$$

| Symbol | Domain/shape | Implemented at |
|--------|--------------|----------------|
| `output_dim = F · (L+1)` | scalar attribute | `src/molzoo/allegro.py:464` |
| `edge_features` | `(E, F·(L+1))` | `src/molzoo/allegro.py:557–558` |

### 5.7 Properties

| Transformation | Property on `edge_features` |
|----------------|-----------------------------|
| `r_i → r_i + t` (translation) | invariant |
| `r_i → R r_i` with `R ∈ O(3)` | invariant¹ |
| Permutation of atoms (global relabelling) | equivariant (edges follow their endpoints) |
| Time reversal / parity | invariant² |

¹ All output channels are L=0 scalars; the tensor track `V` is `O(3)`-equivariant intermediately and is contracted to scalars via the L=0 block at every layer.
² Parity: only scalars are emitted; all odd-parity content is absorbed inside the TP path.

**Locality.** For any pair `(i,j)`, `edge_features[i,j]` depends only on `{k ∈ \mathcal{N}(i)}` via the scattered env weights `\bar{E}_i^{(ℓ)}` (and on `j` via `V_{ij}` itself). Because every layer's TP reads `V_{ij}` (per-edge) and `\bar{E}_i^{(ℓ)}` (scatter at `i` only), the receptive field does not expand with depth — it stays `r_cut`.

**Smoothness at `r_cut`.** Every per-edge quantity descends from `edge_radial[ij] = bessel · u(r_{ij})`, and every Linear in the encoder is bias-free, so `edge_features[ij] → 0` as `r_{ij} → r_cut`. Since `u(r_cut) = 0` together with `u'(r_cut) = 0` for `p ≥ 2`, `edge_features` and its first derivative w.r.t. `r_{ij}` both vanish at the cutoff, giving `C^1`-continuous forces. No additional encoder-level gate is required.

## 6. Config Mapping
- Flag naming differences, default deltas, unit conventions. Missing entries use `—` (never blank).

| `AllegroSpec` field | Reference name | Meaning | Default this/ref | Notes |
|---------------------|----------------|---------|-------------------|-------|
| `num_elements` | `num_types` | Atomic-number embedding table size | `<required>` / dataset-derived | Constraint `> 0`. |
| `num_scalar_features` | `num_scalar_features` | `F` — scalar channel multiplicity | `64` / `64` | Constraint `> 0`. Paper QM9: `1024`. |
| `num_tensor_features` | `num_tensor_features` | `u` — tensor channel multiplicity | `16` / `32` | Constraint `> 0`. Paper QM9: `256`. |
| `r_max` | `r_max` | `r_cut` in Å | `<required>` / dataset-defined | Constraint `> 0`. |
| `num_bessel` | `num_bessels` | Bessel basis count `N_bessel` | `8` / `8` | Constraint `> 0`. |
| `l_max` | `l_max` | `ℓ_max` | `2` / `2` | Constraint `≥ 0`. |
| `num_layers` | `num_layers` | Depth `L` | `2` / `1` | Constraint `> 0`. Paper QM9: `3`. |
| `poly_p` | `polynomial_cutoff_p` | Polynomial cutoff exponent `p` | `6` / `6` | Constraint `≥ 1`. |
| `type_embed_dim` | `module_output_dim` (in `TwoBodyBesselScalarEmbed`) | Type-embed width `d_type` = the entire two-body scalar embed dim (no extra MLP) | `64` / `64` | Must be even (split center/neighbor halves). |
| `latent_mlp_depth` | `allegro_mlp_hidden_layers_depth` | Per-layer latent MLP depth | `2` / `2` | `≥ 0`. Paper QM9 / 3BPA: `3`. Renamed (drop redundant `allegro_mlp_` prefix; matches paper/mir-group's "latent MLP"). |
| `latent_mlp_width` | `allegro_mlp_hidden_layers_width` | Per-layer latent MLP hidden width | `128` / `128` | Paper QM9: `1024`. Renamed (see above). |
| `latent_activation` | `allegro_mlp_nonlinearity` | Activation between hidden layers (`nn.SiLU` / `None` = deep linear) | `nn.SiLU` / `silu` | Paper 3BPA: `null` (deep linear). |
| `avg_num_neighbors` | `avg_num_neighbors` | Dataset-wide aggregate norm `1/√⟨\|N\|⟩` | `<required>` / dataset-computed | Required (no per-node fallback). |

## 7. Benchmark Contract

### 7.1 Reproduction targets
| Dataset | Metric | Paper | This-repo target | Tolerance |
|---------|--------|-------|-------------------|-----------|
| QM9 (U0, eV) | MAE | `<…>` meV | `<…>` meV | `<…>` |
| 3BPA (E, F) | MAE | `<…>` | `<…>` | `<…>` |
| revMD17 | F-MAE | `<…>` | `<…>` | `<…>` |

### 7.2 Engineering benchmarks
Driver: `python benchmarks/run_allegro_bench.py --device cuda` (also collected via `pytest benchmarks/bm_molzoo/bm_allegro.py --benchmark-only` if `pytest-benchmark` is installed).

Reference numbers measured on **Tesla T4** (Turing, 16 GiB) with `bs=32, N̄=18, L=3, l_max=2, F=128, u=32, num_bessel=8` (343 k params total, ~8.4 k edges per batch). Reproduce with:

```
CUDA_VISIBLE_DEVICES=<n> python benchmarks/run_allegro_bench.py \
    --device cuda --num-layers 3 --l-max 2 --num-scalar-features 128 \
    --num-tensor-features 32 --n-graphs 32 --warmup 5 --repeats 30 --scaling
```

| Quantity | Configuration | T4 measured (2026-04-26) | Tested at |
|----------|---------------|--------------------------|-----------|
| Forward time / batch | bs=32 N̄=18 L=3 l_max=2 F=128 u=32 T4 | **5.70 ± 0.01 ms** (0.67 µs / edge) | `benchmarks/bm_molzoo/bm_allegro.py::BMAllegro::test_forward_energy` |
| Forward+backward time / batch (force grads) | same | **13.53 ± 1.08 ms** (1.60 µs / edge) | `…::test_backward_energy` |
| Forward peak memory | same | **97.4 MiB** | `benchmarks/run_allegro_bench.py` (peak via `torch.cuda.max_memory_allocated`) |
| Forward+backward peak memory | same | **199.6 MiB** | same |
| Scaling: forward µs/edge vs E | bs ∈ {4, 16, 32, 64} (E ∈ ~1k, ~4k, ~8.5k, ~17k) | 3.40 → 0.87 → 0.49 → 0.40 µs/edge (saturates ≈0.4 µs/edge ⇒ asymptotically linear in E) | `benchmarks/run_allegro_bench.py --scaling` |

**Complexity.** Two-body embedding: `O(E · (N_bessel·d_type + d_type·F))`. Per-layer aggregation + TP: `O(E · d_sh · u)` for `MakeWeightedChannels` and scatter; `O(E · u · K_CG)` for the per-channel TP, where `K_CG` is the number of admissible `(ℓ_1, ℓ_2, ℓ_3)` paths at `ℓ_max`. Latent MLP: input dim grows linearly with depth (`F·ℓ + u`) so total compute is `O(L^2 · E · F · H)` for hidden width `H`. Memory dominated by tensor track `V ∈ ℝ^{E × d_sh × u}`.

### 7.3 Invariance / equivariance tests
| Symmetry | Test path | Tolerance |
|----------|-----------|-----------|
| Translation | `tests/test_molzoo/test_symmetry.py::TestTranslationInvariance` | rel ≤ 1e-5 |
| Rotation | `tests/test_molzoo/test_symmetry.py::TestRotationEquivariance` | rel ≤ 1e-4 |
| Permutation | `tests/test_molzoo/test_symmetry.py::TestPermutationEquivariance` | rel ≤ 1e-5 |
| Energy translation invariance | `tests/test_molzoo/test_allegro.py::TestEnergyInvariants::test_translation_invariance` | rel ≤ 1e-4 |
| Energy rotation invariance | `tests/test_molzoo/test_allegro.py::TestEnergyInvariants::test_rotation_invariance` | rel ≤ 1e-4 |
| Energy permutation invariance | `tests/test_molzoo/test_allegro.py::TestEnergyInvariants::test_permutation_invariance` | rel ≤ 1e-4 |
| Cutoff vanishing | `tests/test_molzoo/test_allegro.py::TestEnergyInvariants::test_cutoff_vanishing` | rel ≤ 1e-5 |
| Output dim contract `F·(L+1)` | `tests/test_molzoo/test_allegro.py::TestAllegroEncoder::test_output_dim_is_densenet_stack` | exact |
| Edge-feature shape | `tests/test_molzoo/test_allegro.py::TestAllegroEncoder::test_forward_writes_edge_features` | exact |
| Single-batch overfit | `tests/test_molzoo/test_allegro.py::TestOverfitSingleBatch` | loss < 1e-3 |

Tests in §7.3 MUST exist before any §2 row dependent on that symmetry can be marked `matched`.

### 7.4 Run log (append-only)
- One row per benchmark / training run, written by `/molzoo-spec allegro --log ...`.
- Append at the bottom; never reformat earlier rows. `note` may carry a short audit memo (e.g. `⚠️ §2 RBF patched`) backfilled by `molzoo-auditor`.

| run_id | date | commit | dirty | dataset | config | steps | train_mae | val_mae | fwd_ms | bwd_ms | compiled | note |
|--------|------|--------|-------|---------|--------|-------|-----------|---------|--------|--------|----------|------|
| 1 | 2026-04-26 | local | 1 | synthetic-qm9-shape (32×18 atoms, ⟨\|N\|⟩=14.7) | L=3 l_max=2 F=128 u=32 T4 | n/a | n/a | n/a | 5.70 | 13.53 | 0 | engineering bench post-fallback-removal; 0.67 µs/edge fwd, 1.60 µs/edge fwd+bwd, fwd peak 97.4 MiB, fwd+bwd peak 199.6 MiB |

## 8. System Boundary
| Concern | Owner | Contract |
|---------|-------|----------|
| Edge construction | `molix.data.NeighborList` | full bidirectional (`symmetry=True`); `edge_index[:,0]=src/center`, `[:,1]=dst/neighbor`; `bond_diff = pos[dst] − pos[src]`. |
| Encoder forward | `molzoo.allegro.Allegro` | reads §1.1 In; writes §1.1 Out; mutates `GraphBatch` in place. |
| Pair-energy readout | `molpot.heads.EdgeEnergyHead` | reads `("edges","edge_features")`; uses `ScalarMLPFunction` with the same variance-preserving init as the encoder. |
| Energy aggregation | `molpot.heads.EdgeEnergyHead` (scatter to source `/√⟨\|N\|⟩` then `/√2`) + `molpot.PotentialComposer` | owns `("graphs","E")` writes. |
| Force gradients | `molpot.BasePotential.calc_forces` | owns `("atoms","F")`; uses `torch.autograd.grad`. |
| Per-species scale/shift | `molpot` | owns; encoder MUST NOT apply. |
| Training loop | `molix.core.Trainer` | owns `TrainState` namespaces. |

**Hard rules.** Encoder MUST NOT read/write `("graphs",*)`, MUST NOT call `torch.autograd.grad`, MUST NOT mutate input keys, MUST NOT expose the equivariant `V` track. Any change here is a §10.2 breaking change.

## 9. Version Pinning
| Item | Value |
|------|-------|
| Paper | Musaelian et al. 2023, arXiv:2204.05249 |
| Reference repo (Allegro) | `mir-group/allegro` (`allegro/nn/_allegro.py`, `_edgeembed.py`, `scalarembed.py`, `tensorembed.py`, `_strided/_channels.py`, `_strided/_contract.py`, `_edgewise.py`) |
| Reference repo (nequip primitives) | `mir-group/nequip` (`nequip/nn/mlp.py`, `nequip/nn/embedding/_edge.py`) |
| PyTorch | `>= 2.6` |
| cuequivariance / cuequivariance-torch | as pinned in repo |
| This repo commit | tracked by `git log -- src/molzoo/allegro.py` |

## 10. Spec Drift Policy
- **10.1 Triggers — must update spec when:** §5 equation changes → patch §2 row → if behaviour shifts, add §4 row → bump §9; new/renamed config field → §6; §7.1 range moves → §7.4 row + §4 if outside tolerance; reference pin moves → re-audit §3 + refresh §9; touched a file in §2 → verify line resolves, otherwise patch §2.
- **10.2 Breaking — must bump major:** output dict-key rename or shape change (in particular `output_dim` formula); §2 row flips `matched` ⇄ non-`matched`; §4 row removed without replacement; §8 contract change.
- **10.3 Enforcement:** code change without matching `<encoder>.md` diff is reviewable as drift. Run `/molzoo-spec allegro --log` after every benchmark. Run `molzoo-auditor` for any §2 row whose code line numbers no longer resolve.

## Changelog (append-only)
- 2026-04-25 · `253b2c4` · realigned to canonical 10-section template (Scope & Boundary → Spec Drift Policy); preserved equations and test paths verbatim.
- 2026-04-26 · verbatim port of `mir-group/allegro::Allegro_Module` and `nequip` MLP/edge primitives. Output contract changed from `(E, F)` (last-layer scalar) / `(E, L, F)` (per-layer stack) to `(E, F·(L+1))` DenseNet stack; encoder-level final cutoff gate removed (cutoff now propagates implicitly through bias-free linears); α-residual replaced by DenseNet scalar accumulation; separate `env_embed` Linears collapsed into per-layer `latents` MLP output slice; type embedding switched to multiplicative `ProductTypeEmbedding`; `avg_num_neighbors` is now required (no per-node fallback).
- 2026-04-26 (later) · second pass against the reference source. **Architectural fixes:** (a) removed the spurious `scalar_embed_mlp` between `ProductTypeEmbedding` and the layer loop — reference's two-body scalar embed is just `type_embed × basis_linear(bessel)` of dim `type_embed_dim` and feeds `env_embed_linear` / `first_layer_env_embed_projection` directly; (b) fused TP descriptor output segments per `ir3` (`_allegro_uuu_descriptor` now takes `irreps_out` and routes every CG path producing the same ir into a shared output segment) and dropped the post-TP `cuet.Linear`, so trainable weight count matches reference's `(u, num_paths_to_ir)`; (c) added the reference's backward dead-path pruning (`_build_layer_irreps`). **Cleanup:** dropped `# Re-exports for backwards-compatible imports`, trimmed `__all__` to `{Allegro, AllegroSpec}`, removed `scalar_embed_mlp_hidden_layers_{depth,width}` config fields from `AllegroSpec` / `molcfg.yaml` / train scripts / tests, and updated `EdgeEnergyHead` docstring to no longer reference the retired `Allegro(return_all_layers=True)` flag.
- 2026-04-26 (final) · third pass — strict "no fallback" enforcement. **Code-drift fixes (⚠️):** (i) `EdgeEnergyHead.mlp` switched from `nonlinearity=nn.SiLU` back to `nonlinearity=None`, matching `mir-group/allegro::minimal.yaml::edge_eng_mlp_nonlinearity=null` and §2 row "Pair-energy readout MLP (linear, no nonlinearity, no bias)" — readout is purely linear (two `Linear` layers in series); (ii) removed `bond_diff / (bond_dist + 1e-8)` numerical shim in `Allegro.forward` — `SphericalHarmonics(normalize=True)` already normalises internally and `NeighborList` guarantees `bond_dist > 0`, so passing `bond_diff` directly is both correct and one division per edge cheaper. **Fallbacks removed:** dropped `EdgeEnergyHead.avg_num_neighbors=None` per-source path (`_per_source_sqrt_norm` deleted) and `EdgeEnergyHead.num_layers=L` per-layer-MLP variant — the encoder now only writes the flat DenseNet stack `("edges","edge_features")` of shape `(E, F·(L+1))` and the head consumes it as a single feature vector, byte-identical to upstream. `avg_num_neighbors` is a required positional dataset statistic with no fallback. **Tests:** `tests/test_molpot/test_heads/test_edge_energy.py::TestPerLayerReadout` and `test_per_source_fallback_runs` removed; new `TestLinearReadout` asserts `E(2f) = 2 E(f)` and `E(f+g) = E(f) + E(g)` to lock in the linear-readout invariant.
