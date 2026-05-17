#!/usr/bin/env bash
#SBATCH --job-name=ticketflow-a40-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/a40_ticketflow_eval_%j.out
#SBATCH --error=logs/a40_ticketflow_eval_%j.err

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
python -m pip install --upgrade pip setuptools wheel
python - <<'PY' || python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
python -m pip install -e ".[dev,rag-pro]"

echo "== Smoke tests =="
pytest tests/test_ticketflow.py::test_hybrid_retriever_records_bm25_rrf_and_feature_rerank \
       tests/test_ticketflow.py::test_retrieval_metrics_report_hit_rate_mrr_and_error_rate \
       tests/test_ticketflow.py::test_history_support_strength_uses_calibrated_feature_signals -q

echo "== 600-sample retrieval benchmark: BM25 + BGE-M3 + RRF + BGE-Reranker =="
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
python -m ticketflow.eval_retrieval_strategies \
  --project-root "$PROJECT_ROOT" \
  --sample-size 600 \
  --report-path reports/retrieval_strategy_eval_a40_bge_v6.json

echo "== Cloud LLM connectivity =="
export MODEL_BACKEND=cloud_api
python -m ticketflow.llm_check --project-root "$PROJECT_ROOT" --target cloud

echo "== RAG-sensitive cloud benchmark =="
export RAG_CACHE_ENABLED="${RAG_CACHE_ENABLED:-true}"
python -m ticketflow.eval_rag_sensitive \
  --project-root "$PROJECT_ROOT" \
  --eval-csv data/gold_eval/rag_sensitive_eval_v2.csv \
  --report-path reports/rag_sensitive_eval_cloud_a40_v6.json \
  --cache-version rag_sensitive_cloud_a40_v6 \
  --batch-size 5 \
  --clear-cache

echo "== Done =="
date
