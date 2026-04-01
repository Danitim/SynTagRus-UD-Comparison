#!/bin/bash
set -euo pipefail

# Usage: ./predict_udpipe2.sh [mBERT|ModernBERT|all]
MODEL_TAG="${1:-all}"
case "$MODEL_TAG" in
  mBERT|ModernBERT|all) ;;
  *)
    echo "Usage: $0 {mBERT|ModernBERT|all}"
    exit 1
    ;;
esac

CORPORA=(ud ud-new ud-old str str-new str-old)
SEEDS=(7 91 333 678 1999)
NUM_RUNS=${#SEEDS[@]}
PYTHON="${PYTHON:-python}"
UDPIPE2="vendor/udpipe2/udpipe2.py"

if [[ ! -f "$UDPIPE2" ]]; then
  echo "[ERR] File not found: $UDPIPE2"
  exit 1
fi

# ── ENV ──────────────────────────────────────────────────────────────────────
if [[ ! -d ".venv-udpipe" ]]; then
  echo "=== [ENV] Creating .venv-udpipe via uv ==="
  pip install -q uv
  uv venv .venv-udpipe
  source .venv-udpipe/bin/activate
  uv pip install "tensorflow>=2.3.1" "transformers>=4,<5" ufal.chu_liu_edmonds ufal.udpipe
else
  echo "=== [ENV] Using existing .venv-udpipe ==="
  source .venv-udpipe/bin/activate
fi

# ── PREDICT FUNCTION ─────────────────────────────────────────────────────────
predict_run () {
  local model_dir="$1"
  local in_file="$2"
  local out_file="$3"

  if [[ ! -d "$model_dir" ]]; then
    echo "[SKIP] Model not found: ${model_dir}"
    return
  fi
  if [[ ! -f "$in_file" ]]; then
    echo "[SKIP] Input not found: ${in_file}"
    return
  fi

  mkdir -p "$(dirname "$out_file")"
  echo "[RUN ] ${model_dir}"
  echo "       ${in_file} -> ${out_file}"

  $PYTHON "$UDPIPE2" "$model_dir" \
    --predict \
    --predict_input "$in_file" \
    --predict_output "$out_file"

  if [[ -s "$out_file" ]]; then
    echo "[OK  ] ${out_file}"
  else
    echo "[WARN] Empty output: ${out_file}"
  fi
}

# ── mBERT ─────────────────────────────────────────────────────────────────────
run_mbert () {
  echo "=== [mBERT] Running predictions ==="
  for corpus in "${CORPORA[@]}"; do
    for run_idx in $(seq 1 "$NUM_RUNS"); do
      predict_run \
        "models/UDPipe2/mBERT/${corpus}-run${run_idx}" \
        "datasets/${corpus}/test.conllu" \
        "out/UDPipe2/mBERT/run${run_idx}/${corpus}/test.pred.conllu"
    done
  done
  echo "=== [mBERT] Done ==="
}

# ── ModernBERT ────────────────────────────────────────────────────────────────
run_modernbert () {
  echo "=== [ModernBERT] Running predictions ==="
  for run_idx in $(seq 1 "$NUM_RUNS"); do
    for corpus in "${CORPORA[@]}"; do
      predict_run \
        "models/UDPipe2/ModernBERT/run${run_idx}/${corpus}" \
        "datasets_modern/${corpus}/test.conllu" \
        "out/UDPipe2/ModernBERT/run${run_idx}/${corpus}/test.pred.conllu"
    done
  done
  echo "=== [ModernBERT] Done ==="
}

# ── DISPATCH ──────────────────────────────────────────────────────────────────
case "$MODEL_TAG" in
  mBERT)      run_mbert ;;
  ModernBERT) run_modernbert ;;
  all)        run_mbert; run_modernbert ;;
esac

echo "=== [ALL DONE] Predictions are in out/UDPipe2/{mBERT,ModernBERT}/ ==="
