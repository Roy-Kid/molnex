---
slug: sonata-05-obs
criteria:
  - id: ac-001
    summary: RDF on FCC lattice recovers analytic peak positions within 1%
    type: code
    pass_when: |
      tests/test_molix/test_observables/test_rdf.py::test_rdf_fcc_lattice_peaks
      passes; on a 4x4x4 FCC supercell with lattice constant a=4.0 Å under PBC,
      RDF(cutoff=6.0, n_bins=600, pair_types=[(1,1)]).result() yields g(r) local
      maxima at r1=a/√2, r2=a, r3=a·√(3/2) with bin-centre error < 0.01 Å each.
    status: pending

  - id: ac-002
    summary: DipoleDipoleACF recovers exponential decay constant within 5%
    type: code
    pass_when: |
      tests/test_molix/test_observables/test_correlation.py::test_acf_exp_decay
      passes; for a synthetic series μ(t)=μ_0·exp(-t/τ)+ε with τ=0.5 ps, dt=1 fs,
      N=1000 steps, the recovered τ from fitting ln(ACF(τ)) is within 5% of 0.5 ps.
    status: pending

  - id: ac-003
    summary: MultipoleStability scalar stats match closed-form expectations on fixture
    type: code
    pass_when: |
      tests/test_molix/test_observables/test_multipole_stability.py::test_stats_on_deterministic_fixture
      passes; for a 100-frame deterministic fixture with q_perm sampled from a
      known distribution (mean 0, std 0.1 e), MultipoleStability.result()
      reports mean, std, min, max each within numerical tolerance (1e-6) of
      the analytic values.
    status: pending

  - id: ac-004
    summary: run_obs_stage smoke run on 10-frame synthetic trajectory writes all artifacts
    type: runtime
    pass_when: |
      tests/test_molix/test_observables/test_obs_driver.py::test_obs_driver_smoke
      passes; driver consumes a 10-frame synthetic trajectory.xyz + matching
      BenchManifest, returns the artifact dir Path, and that dir contains
      observables.json, rdf.npz, dipole_acf.npz, multipole_stability.json
      (Sonata branch) or rdf.npz + skipped-stub artifacts (Allegro branch).
    status: pending

  - id: ac-005
    summary: O-O first peak position from a real Sonata MD run lies in [2.73, 2.78] Å ±0.05 Å
    type: scientific
    pass_when: |
      A Sonata-MD run on bulk water (RPBE-D3 manifest), after running
      `pytest -m bench_obs tests/test_molix/test_observables/test_obs_driver.py::test_oo_peak_real_md`,
      yields rdf.npz whose O-O g(r) first maximum (argmax over r in [2.5, 3.2] Å)
      falls within [2.68, 2.83] Å. Tolerance ±0.05 Å is engineering judgment,
      not literature-pinned; documented in test docstring.
    status: pending

  - id: ac-006
    summary: Sonata vs Allegro dipole ACFs are distinguishable beyond noise floor at τ=1 ps
    type: scientific
    pass_when: |
      Running the obs driver against the same MD trajectory once under a
      Sonata manifest and once under an Allegro manifest with
      `pytest -m bench_obs tests/test_molix/test_observables/test_obs_driver.py::test_acf_lr_vs_sr`,
      the absolute difference |ACF_sonata(1 ps) - ACF_allegro(1 ps)| exceeds
      2× the per-curve noise floor (std of ACF tail over τ in [2 ps, max_lag]).
      Allegro branch's ACF here comes from a control synthesis (no q/μ
      channel ⇒ uses positional dipole proxy ΣZ_i·r_i) to provide a non-trivial
      baseline, documented in the test.
    status: pending

  - id: ac-007
    summary: MultipoleStability bounds hold over a full Sonata run
    type: scientific
    pass_when: |
      Running `pytest -m bench_obs tests/test_molix/test_observables/test_obs_driver.py::test_multipole_bounds_sonata`
      on a Sonata-MD run, multipole_stability.json reports |Σq|/N_atoms < 1e-4,
      max|q_perm| < 5.0 e, and E_elec_over_E_short_mean < 1.0. All three
      bounds are engineering, not literature-pinned, and the test docstring
      cites them as such.
    status: pending

  - id: ac-008
    summary: Lint and full test suite pass
    type: runtime
    pass_when: |
      `ruff check src/molix/observables src/molix/bench/drivers/obs.py
       tests/test_molix/test_observables` exits 0 AND
      `python -m pytest tests/test_molix/test_observables -v` exits 0
      (bench_obs-marked tests excluded from the default suite).
    status: pending
---

# Acceptance criteria

The eight criteria above bind `sonata-05-obs` to its observable computational kernels (ac-001..003), driver wiring (ac-004), the load-bearing physics claims of the Sonata chain (ac-005..007: O-O peak position, LR-vs-SR dipole-correlation distinguishability per Cheng 2025 Fig 4, and multipole-stability engineering bounds), and the standard repo cleanliness gate (ac-008). Criteria ac-005, ac-006, ac-007 are gated behind the `bench_obs` pytest marker so the default test suite stays fast; they execute only when a real Sonata MD run is staged under the `runs/` layout from earlier sub-specs.
