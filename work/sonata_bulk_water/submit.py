"""Alvis (C3SE) sbatch renderer for the Sonata bulk-water trainer.

Renders an sbatch script via :class:`string.Template`, writes it under
``<out_dir>/submit.sbatch``, and optionally submits it via ``sbatch``.
Only the Alvis ``alvis`` partition + A100 GPU shape is supported; other
clusters require a separate driver.

Usage::

    python work/sonata_bulk_water/submit.py \\
        --account NAISS2025-X-YYY \\
        --time 24:00:00 \\
        --data-root /mimer/.../water_les \\
        --out-dir $HOME/runs/sonata_water_$(date +%Y%m%d) \\
        --max-epochs 100 --batch-size 4 \\
        --gpu A100             # or A100fat for 80 GB

    # rendering check (no sbatch call):
    python work/sonata_bulk_water/submit.py … --dry-run

References:
    C3SE Alvis user manual — partition ``alvis``, GPU shapes
    ``A100:1`` (40 GB) and ``A100fat:1`` (80 GB), project accounts
    ``NAISS<YYYY>-<X>-<Z>``.
    https://www.c3se.chalmers.se/documentation/for_users/intro-alvis/
"""

from __future__ import annotations

import argparse
import re
import string
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults the user should review on Alvis login node
# ---------------------------------------------------------------------------

# Default module load. Two cases the contributor picks between:
#
# (a) **Self-contained venv** (e.g. uv-installed PyTorch wheels with bundled
#     cuXXX libs). The molnex C++ extension ``libmolnex_opLib.so`` still
#     links against system ``libnvrtc.so.12``, so the compute node must
#     expose at least ``CUDA/12.x`` after ``module purge``. ``CUDA/12.6.0``
#     is the load on Alvis at spec time.
# (b) **EasyBuild PyTorch** (no venv, or venv is a thin layer on top).
#     Pass ``--module PyTorch/2.6.0-foss-2024a-CUDA-12.1.1`` (or whatever
#     ``module avail PyTorch`` lists). PyTorch's EasyBuild module pulls
#     CUDA in as a dependency, so ``CUDA/`` does not need to be loaded
#     separately.
#
# Pass ``--module ""`` to skip the load entirely (only safe when the
# venv really is self-contained and the C++ extension was built against
# a CUDA already on the system loader path).
DEFAULT_MODULE = "CUDA/12.6.0"

# Virtual-environment directory inside the user's portfolio. Override
# via ``--venv`` (the user is expected to have created the venv on the
# login node before first submission — see the README's "Alvis setup"
# paragraph).
DEFAULT_VENV = "$HOME/portfolio/venvs/molnex"

# Per-driver constants.
DEFAULT_PARTITION = "alvis"
DEFAULT_GPU = "A100"  # 40 GB; switch to "A100fat" for 80 GB
DEFAULT_JOB_NAME = "sonata_water"


SBATCH_TEMPLATE = string.Template(
    """#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH --gpus-per-node=${GPU}:1
#SBATCH -t ${TIME}
#SBATCH -J ${JOB_NAME}
#SBATCH -o ${OUT_DIR}/slurm-%j.out
#SBATCH -e ${OUT_DIR}/slurm-%j.err

set -euo pipefail
module purge
${MODULE_LOAD}
source ${VENV}/bin/activate

cd ${REPO_ROOT}
python -u work/sonata_bulk_water/train.py \\
    --data-root ${DATA_ROOT} \\
    --out-dir ${OUT_DIR} \\
    --max-epochs ${MAX_EPOCHS} \\
    --batch-size ${BATCH_SIZE} \\
    --lr ${LR} \\
    --seed ${SEED}
"""
)


def _repo_root() -> Path:
    """Project root — two levels above this script (work/sonata_bulk_water/)."""
    return Path(__file__).resolve().parent.parent.parent


def _render(args: argparse.Namespace) -> str:
    """Render SBATCH_TEMPLATE with the parsed CLI args."""
    # Empty ``--module`` skips ``module load`` entirely. This is the right
    # path when the activated venv already bundles a self-contained PyTorch
    # CUDA stack (uv-installed wheels with cuXXX libs) and a system EasyBuild
    # PyTorch module would only introduce a CUDA-version conflict.
    module_load = (
        f"module load {args.module}" if args.module else "# (no module load — venv self-contained)"
    )
    return SBATCH_TEMPLATE.substitute(
        ACCOUNT=args.account,
        PARTITION=args.partition,
        GPU=args.gpu,
        TIME=args.time,
        JOB_NAME=args.job_name,
        OUT_DIR=str(args.out_dir),
        MODULE_LOAD=module_load,
        VENV=args.venv,
        REPO_ROOT=str(_repo_root()),
        DATA_ROOT=str(args.data_root),
        MAX_EPOCHS=args.max_epochs,
        BATCH_SIZE=args.batch_size,
        LR=args.lr,
        SEED=args.seed,
    )


def _parse_argv(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render and (optionally) submit the Sonata Alvis sbatch script.",
    )
    p.add_argument("--account", required=True, help="SLURM account, e.g. NAISS2025-X-YYY.")
    p.add_argument("--time", required=True, help="SLURM walltime, e.g. 24:00:00.")
    p.add_argument("--data-root", type=Path, required=True, help="WaterLES dataset root.")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for the run.")
    p.add_argument("--partition", default=DEFAULT_PARTITION, help="SLURM partition.")
    p.add_argument(
        "--gpu",
        default=DEFAULT_GPU,
        help="Alvis GPU shape: A100 (40 GB) or A100fat (80 GB).",
    )
    p.add_argument("--job-name", default=DEFAULT_JOB_NAME, help="SLURM job name.")
    p.add_argument("--module", default=DEFAULT_MODULE, help="EasyBuild module to load.")
    p.add_argument(
        "--venv", default=DEFAULT_VENV, help="Virtualenv directory (bin/activate sourced)."
    )
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the rendered sbatch to stdout; do not call sbatch.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_argv(argv)
    rendered = _render(args)

    if args.dry_run:
        sys.stdout.write(rendered)
        return 0

    # Non-dry-run: write the script, then submit.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    submit_path = args.out_dir / "submit.sbatch"
    submit_path.write_text(rendered)
    proc = subprocess.run(
        ["sbatch", str(submit_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    match = re.search(r"Submitted batch job (\d+)", proc.stdout)
    if match is None:
        sys.stderr.write("submit.py: could not parse jobid from sbatch stdout\n")
        return 1
    jobid = match.group(1)
    (args.out_dir / "jobid.txt").write_text(jobid)
    sys.stdout.write(f"jobid={jobid} written to {args.out_dir / 'jobid.txt'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
