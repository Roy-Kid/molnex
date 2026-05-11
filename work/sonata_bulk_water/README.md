# Sonata bulk-water RPBE-D3 — Alvis A100 training driver

Two-script training entry point for the Sonata long-range MLIP on bulk
liquid water (RPBE-D3, 64-H₂O periodic cubic box). Runs as a single
process on one A100; no DDP, no rank-0 guards. Reproduces the paper
scale of Cheng B., *npj Comput. Mater.* **11**:80 (2025).

## Quick start

```bash
# Local smoke (CPU or single GPU):
python -u work/sonata_bulk_water/train.py \
    --data-root ~/datasets/water_les \
    --out-dir /tmp/sonata_smoke \
    --max-epochs 1 --batch-size 2 --lr 1e-3 --seed 0

# Submit a full run on Alvis (login node):
python work/sonata_bulk_water/submit.py \
    --account NAISS2025-X-YYY \
    --time 24:00:00 \
    --data-root /mimer/.../water_les \
    --out-dir $HOME/runs/sonata_water_$(date +%Y%m%d) \
    --max-epochs 100 --batch-size 4 \
    --gpu A100        # or A100fat for 80 GB
```

After training, `<out-dir>` holds:

- `metrics.json` — final test-set MAE/RMSE (`energy_mae_meV_per_atom`,
  `force_rmse_meV_per_A`).
- `tb/` — TensorBoard event files.
- `journal/` — append-only Zarr v3 store written by `JournalHook` →
  `JournalWriter` (one record per `train/`, `eval/`, `performance/`,
  `gpu/` scalar at every 10 training steps and every eval phase).
- `checkpoints/last.pt`, `checkpoints/best.pt` — `CheckpointHook` output
  (best is gated on `eval/ForceRMSE`).
- `train.log` — stderr-mirrored INFO log.

## Alvis setup

The driver is the SLURM-submission piece; you still have to lay down a
Python environment on the login node before the first submission:

1. `module avail PyTorch` — list the EasyBuild modules. Pick one with
   the CUDA version matching A100s currently in your reservation; pass
   it via `--module` if the script's default
   (`PyTorch/2.6.0-foss-2024a-CUDA-12.1.1`) has rolled off the system.
2. `python -m venv $HOME/portfolio/venvs/molnex && \
   source $HOME/portfolio/venvs/molnex/bin/activate && \
   pip install -e .[dev]` from the repo root. Override the venv path
   via `--venv` if you keep your environments elsewhere.
3. Account: pick from `projinfo`; format `NAISS<YYYY>-<X>-<Z>`.
4. GPU shape: `--gpu A100` for 40 GB cards, `--gpu A100fat` for 80 GB.
   The 64-H₂O box at `batch_size=4` fits comfortably in 40 GB; raise
   to `A100fat` only when scaling batch / box size.

## Paper hyperparameter provenance

Every Sonata hyperparameter in `train.py` is grep-locked to its
canonical value. The source of truth:

- Encoder shape (`r_max=5.0`, `l_max=2`, `num_features=64`,
  `num_layers=2`, `type_embed_dim=32`, `latent_mlp_width=64`,
  `avg_num_neighbors=12.0`): `benchmarks/bm_molpot/bm_sonata.py:51-95`
  (the Sonata branch of `_build_sonata_and_baseline`), itself matching
  Cheng B. 2025 §IV defaults.
- Ewald parameters (`sigma=1.0 Å`, `dl=2.0 Å`): Cheng B.,
  *Latent Ewald summation for machine-learning potentials*,
  **npj Comput. Mater.** 11:80 (2025), §III defaults.
  [doi:10.1038/s41524-025-01577-7](https://doi.org/10.1038/s41524-025-01577-7).
- Force-RMSE baseline (~32.1 meV·Å⁻¹) for RPBE-D3 liquid water:
  *J. Chem. Phys.* **163**:104102 (2025), Table 1.

If a numeric value needs to change, update **both** the constant in
`train.py` *and* the citation here. `ac-005` of `sonata-05-run` is the
grep gate that catches silent drift.

## NaN early-stop semantics

`NaNStopHook` checks `state["train"]["loss"]` and every model parameter
on every train-batch end. On detection:

1. `<out_dir>/nan_checkpoint.pt` — the model state-dict at the abort step.
2. `train.log` final line contains `"NaN detected"`.
3. Process exit code `2` (not `1`). This lets SLURM
   `--mail-type=FAIL` and any outer wrapper distinguish "Ewald blew up"
   from "ImportError / OOM / other unhandled exception" (which exits `1`).

To exercise the path without a real NaN, pass `--debug-inject-nan` — a
debug-only hook overwrites `state["train"]["loss"]` with `NaN` at step 2.
Used by `ac-002`; never pass in production.

## Out of scope

- DDP / FSDP / multi-GPU — a single A100 is enough at paper scale.
- Hydra / OmegaConf — argparse + paper defaults is sufficient; matches
  `bm_sonata.py`'s ergonomics.
- Hyperparameter sweep — one `submit.py` call submits one job;
  multi-job campaigns are an external bash loop's responsibility.
- Other clusters (Berzelius, Tetralith, …) — `submit.py` is Alvis-only
  by design.
