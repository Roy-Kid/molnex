---
slug: sonata-05-md
criteria:
  - id: ac-001
    summary: NVE integrator conserves energy on harmonic toy to atol 1e-6
    type: code
    pass_when: |
      tests/test_molix/test_md/test_integrator_nve.py::test_nve_harmonic_energy_conservation
      passes; running VelocityVerletNVE for 1000 steps at dt=0.5 fs in float64
      on a 3-atom isotropic harmonic toy potential keeps
      |E(t) - E(0)| / |E(0)| < 1e-6 at every logged frame.
    status: pending
  - id: ac-002
    summary: Minimum-image wrap is idempotent and keeps atoms inside cell
    type: code
    pass_when: |
      tests/test_molix/test_md/test_integrator_nve.py::test_minimum_image_idempotent
      passes; after one VelocityVerletNVE.step on an orthorhombic 64-water box
      that intentionally drifts an atom 1.5 box-lengths, positions satisfy
      0 <= pos < diag(cell) and applying _wrap_into_cell again leaves pos
      bitwise-identical.
    status: pending
  - id: ac-003
    summary: Langevin BAOAB reaches target temperature within 5 % over 50 ps
    type: code
    pass_when: |
      tests/test_molix/test_md/test_thermostat_langevin.py::test_baoab_target_temperature
      passes; coupling a 256-DOF harmonic toy to LangevinThermostat(T_target=300 K,
      gamma=1.0/ps) at dt=0.5 fs for 100000 steps yields a sample-mean
      instantaneous temperature over the last 50 ps within
      [285 K, 315 K].
    status: pending
  - id: ac-004
    summary: molpy XYZTrajectoryWriter round-trip preserves Z, pos, cell
    type: code
    pass_when: |
      tests/test_molix/test_md/test_io_xyz_roundtrip.py::test_xyz_writer_reader_roundtrip
      passes; writing a 64-water MDState via mdstate_to_molpy_frame +
      XYZTrajectoryWriter then reading back recovers Z exactly, |pos_read -
      pos_written|_max < 1e-4 Å, and cell (parsed from "Lattice=...") matches
      to atol=1e-6 Å.
    status: pending
  - id: ac-005
    summary: run_md_stage smoke runs 100 NVE steps on water fixture without NaN
    type: runtime
    pass_when: |
      pytest tests/test_molix/test_bench/test_md_driver.py::test_run_md_stage_smoke
      exits 0; the test builds a tiny Sonata-like model, points BenchManifest
      at the fixture checkpoint, runs run_md_stage for n_steps=100 at dt=0.5 fs,
      and asserts trajectory.xyz and md_log.json exist under
      manifest.to_artifact_dir(), md_log has 100/write_every entries, every
      e_total / temperature / max_force is finite, and nan_count == 0 on all
      frames.
    status: pending
  - id: ac-006
    summary: Side-tier isolation — molix.md is not imported by training code
    type: code
    pass_when: |
      tests/test_molix/test_md/test_side_tier_isolation.py::test_core_does_not_import_md
      passes; after a fresh `importlib.import_module("molix.core")` in a
      subprocess, `"molix.md"` is absent from sys.modules; and importing
      molix.md does not pull in molix.core.trainer.
    status: pending
  - id: ac-007
    summary: Nightly Sonata-MD bar — 25 ps NVE + 50 ps NVT on real water
    type: performance
    evaluator_hint: nightly bench_md marker
    pass_when: |
      pytest -m bench_md tests/test_molix/test_bench/test_md_bench.py::test_sonata_md_nightly
      exits 0; using a trained sonata-05-train checkpoint and
      WaterLESSource(split="test")[0], a 25 ps NVE (50000 steps @ 0.5 fs) shows
      max |E_total(t) - E_total(0)| / (N_atoms * t_ps) < 10 meV/atom/ps,
      NaN count over the full trajectory == 0, and instantaneous max|F| stays
      < 50 eV/Å on every logged frame. NB: drift and force bars are
      engineering rules of thumb, not literature-pinned for RPBE-D3 64-water.
    status: pending
  - id: ac-008
    summary: NVT temperature within ±15 % of target over last 25 ps
    type: scientific
    pass_when: |
      In the same nightly run as ac-007, after a 50 ps NVT segment at
      T_target=300 K (Langevin gamma=1.0/ps), the time-average of
      md_log.temperature_K over the final 25 ps lies in [255 K, 345 K]
      (i.e. ±15 % of T_target). Sanity bound, not literature-pinned.
    status: pending
  - id: ac-009
    summary: Lint + full test suite green
    type: runtime
    pass_when: |
      `python -m pytest tests/ -v` exits 0 (excluding -m bench_md which is
      nightly-only), and the project's standard lint pass (ruff / format)
      reports no new findings on files added under src/molix/md/ and
      src/molix/bench/drivers/.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002** lock Velocity-Verlet correctness on a toy harmonic potential, decoupled from any real MLIP — this is the cheapest, sharpest test of integrator soundness and runs in milliseconds.
- **ac-003** validates the BAOAB Langevin scheme against the equipartition theorem (mean kinetic energy ↔ target T) on a known-analytic system; a 5 % envelope is tight enough to catch one-sided drift in the OU half-steps but loose enough not to flake on stochastic noise at 100k steps.
- **ac-004** ensures the molpy XYZ writer/reader pair (the only IO this stage commits to) round-trips the three fields the observables stage needs: `Z`, `pos`, `cell`. Forces / energies live in `md_log.json` by design.
- **ac-005** is the integration smoke: a tiny model + fixture checkpoint + 100 steps. No physics bar here — only "does the driver actually produce both artifacts and avoid NaN propagation?"
- **ac-006** is the architectural guard: `molix.md` is a side-tier (like `molix.profiler`), so training code must not transitively import it. Enforced by sys.modules introspection in a subprocess.
- **ac-007 / ac-008** are the real-physics nightly bar, gated behind `pytest.mark.bench_md`. Bars are explicitly engineering rules of thumb (< 10 meV/atom/ps drift, < 50 eV/Å max-force, ±15 % T) and the spec text and pass_when both call this out — no false claim of literature-pinned thresholds for RPBE-D3 64-water.
- **ac-009** is the standard end-of-task hygiene check.
