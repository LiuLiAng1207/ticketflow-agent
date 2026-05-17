#!/usr/bin/env bash
#SBATCH --job-name=ticketflow-rag-v11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/a40_rag_sensitive_v11_%j.out
#SBATCH --error=logs/a40_rag_sensitive_v11_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
BASE_PYTHON="${BASE_PYTHON:-$HOME/conda_envs/rl-zvp-repro/bin/python}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv-a40}"

cd "$PROJECT_ROOT"
mkdir -p logs reports

if [ ! -x "$BASE_PYTHON" ]; then
  echo "Cannot find BASE_PYTHON=$BASE_PYTHON." >&2
  exit 1
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "== Runtime =="
hostname
date
nvidia-smi || true
python --version

echo "== Install editable project =="
python -m pip install -e ".[dev,rag-pro]"

echo "== Smoke tests =="
pytest tests/test_ticketflow.py::test_history_support_strength_uses_calibrated_feature_signals \
       tests/test_ticketflow.py::test_llm_sufficiency_cannot_mark_present_required_source_missing \
       tests/test_ticketflow.py::test_same_ticket_history_is_prioritized_as_thread_memory \
       tests/test_ticketflow.py::test_refund_context_keeps_multiple_policy_rules_when_available -q

echo "== RAG-sensitive cloud benchmark v11 =="
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RAG_EMBED_MODEL="${RAG_EMBED_MODEL:-BAAI/bge-m3}"
export RAG_EMBED_BACKEND="${RAG_EMBED_BACKEND:-transformers}"
export RAG_RERANKER_BACKEND="${RAG_RERANKER_BACKEND:-bge}"
export RAG_RERANKER_MODEL="${RAG_RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}"
export RAG_RERANKER_DEVICE="${RAG_RERANKER_DEVICE:-cuda:0}"
export RAG_TOP_K="${RAG_TOP_K:-6}"
export MODEL_BACKEND=cloud_api
export RAG_CACHE_ENABLED="${RAG_CACHE_ENABLED:-true}"

python -m ticketflow.llm_check --project-root "$PROJECT_ROOT" --target cloud
python -m ticketflow.eval_rag_sensitive \
  --project-root "$PROJECT_ROOT" \
  --eval-csv data/gold_eval/rag_sensitive_eval_v2.csv \
  --report-path reports/rag_sensitive_eval_cloud_a40_v11.json \
  --cache-version rag_sensitive_cloud_a40_v11 \
  --batch-size 5 \
  --clear-cache

echo "== Done =="
date
