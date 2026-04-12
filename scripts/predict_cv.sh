#!/bin/bash
set -euo pipefail

MODEL_TAG="${1:-mmBERT}"
DATASETS_ROOT="${2:-datasets_mmbert}"

CORPORA=(
  str-new-cv1 str-new-cv2 str-new-cv3 str-new-cv4
  str-new-cv5 str-new-cv6 str-new-cv7 str-new-cv8 str-new-cv9
  ud-new-cv1  ud-new-cv2  ud-new-cv3  ud-new-cv4
  ud-new-cv5  ud-new-cv6  ud-new-cv7  ud-new-cv8  ud-new-cv9
)
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
