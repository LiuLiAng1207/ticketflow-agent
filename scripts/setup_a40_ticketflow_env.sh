#!/usr/bin/env bash
set -euo pipefail

BASE_PYTHON="${BASE_PYTHON:-$HOME/conda_envs/rl-zvp-repro/bin/python}"
VENV_DIR="${VENV_DIR:-$PWD/.venv-a40}"

if [ ! -x "$BASE_PYTHON" ]; then
  echo "Cannot find BASE_PYTHON=$BASE_PYTHON." >&2
  exit 1
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python - <<'PY' || python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
python -m pip install -e ".[dev,rag-pro]"

python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
