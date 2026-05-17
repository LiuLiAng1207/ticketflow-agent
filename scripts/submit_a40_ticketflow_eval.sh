#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
BASE_PYTHON="${BASE_PYTHON:-$HOME/conda_envs/rl-zvp-repro/bin/python}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv-a40}"

cd "$PROJECT_ROOT"
mkdir -p logs reports

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Please run this on the Slurm login node." >&2
  exit 1
fi

export PROJECT_ROOT
export BASE_PYTHON
export VENV_DIR
sbatch scripts/a40_ticketflow_eval.sbatch
