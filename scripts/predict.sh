#!/bin/bash
set -euo pipefail

MODEL_TAG="${1:-}"
if [[ -z "$MODEL_TAG" ]]; then
  echo "Usage: $0 <MODEL_TAG> [DATASETS_ROOT]"
  echo "  MODEL_TAG:     e.g. mBERT, mmBERT, ModernBERT"
  echo "  DATASETS_ROOT: datasets dir (default: datasets_modern for ModernBERT, datasets otherwise)"
  exit 1
fi

if [[ "$MODEL_TAG" == "ModernBERT" ]]; then
  DATASETS_ROOT="${2:-datasets_modern}"
elif [[ "$MODEL_TAG" == "mmBERT" ]]; then
  DATASETS_ROOT="${2:-datasets_mmbert}"
else
  DATASETS_ROOT="${2:-datasets}"
fi

CORPORA=(ud ud-new ud-old str str-new str-old)
NUM_RUNS=5
PYTHON="${PYTHON:-python}"

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

UDPIPE2="vendor/udpipe2/udpipe2.py"

if [[ ! -f "$UDPIPE2" ]]; then
  echo "[ERR] File not found: $UDPIPE2"
  exit 1
fi

echo "=== [PREDICT] model=${MODEL_TAG} datasets=${DATASETS_ROOT} ==="

for corpus in "${CORPORA[@]}"; do
  for run_idx in $(seq 1 "$NUM_RUNS"); do
    model_dir="models/UDPipe2/${MODEL_TAG}/run${run_idx}/${corpus}"
    in_file="${DATASETS_ROOT}/${corpus}/test.conllu"
    out_file="out/UDPipe2/${MODEL_TAG}/run${run_idx}/${corpus}/test.pred.conllu"

    if [[ ! -d "$model_dir" ]]; then
      echo "[SKIP] Model not found: ${model_dir}"
      continue
    fi
    if [[ ! -f "$in_file" ]]; then
      echo "[SKIP] Input not found: ${in_file}"
      continue
    fi

    mkdir -p "$(dirname "$out_file")"
    echo "[RUN ] run${run_idx}/${corpus}: ${in_file} -> ${out_file}"

    $PYTHON "$UDPIPE2" "$model_dir" \
      --predict \
      --predict_input "$in_file" \
      --predict_output "$out_file"

    if [[ -s "$out_file" ]]; then
      echo "[OK  ] ${out_file}"
    else
      echo "[WARN] Empty output: ${out_file}"
    fi
  done
done

echo "=== [DONE] Predictions are in out/UDPipe2/${MODEL_TAG}/ ==="
