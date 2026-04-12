#!/bin/bash
set -euo pipefail

# 0. Ensure uv is installed and on PATH
if ! command -v uv >/dev/null 2>&1; then
  echo "[BOOT] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$PATH"

echo "[INFO] uv path: $(command -v uv || echo 'NOT FOUND')"
echo "[INFO] python default: $(command -v python)"

# 1. Set up wembeddings venv
echo "=== [STEP 1] Setting up uv venv for wembeddings ==="

if [[ -d ".venv-wemb-mmbert-cv" ]]; then
  echo "[INFO] Existing .venv-wemb-mmbert-cv found, reusing it"
else
  uv venv --python 3.12 .venv-wemb-mmbert-cv
fi

source .venv-wemb-mmbert-cv/bin/activate

uv pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu
if [[ -f vendor/udpipe2/wembedding_service/requirements.txt ]]; then
  uv pip install -r vendor/udpipe2/wembedding_service/requirements.txt
else
  uv pip install numpy scipy tqdm
fi
uv pip install "transformers>=4.48.0" safetensors

echo "[STEP 1] venv ready."

# 2. Compute wembeddings for test sets (skip if .npz already exists)
echo "=== [STEP 2] Computing wembeddings for test sets ==="

WEMB_PY=".venv-wemb-mmbert-cv/bin/python"
WEMB_MODEL="mmBERT-base-last4"
WEMB_MODEL_TAG="mmBERT"
MASK_TOKEN="<mask>"

CV_CORPORA=(
  str-new-cv1 str-new-cv2 str-new-cv3 str-new-cv4
  str-new-cv5 str-new-cv6 str-new-cv7 str-new-cv8 str-new-cv9
  ud-new-cv1  ud-new-cv2  ud-new-cv3  ud-new-cv4
  ud-new-cv5  ud-new-cv6  ud-new-cv7  ud-new-cv8  ud-new-cv9
)

compute_wemb() {
  local inpath="$1"
  local outpath="${inpath}.npz"

  if [[ ! -f "$inpath" ]]; then
    echo "[warn] $inpath not found, skipping" >&2
    return 0
  fi
  if [[ -f "$outpath" ]]; then
    echo "[skip] already have $outpath"
    return 0
  fi

  echo "[WEMB] $inpath"
  "$WEMB_PY" vendor/udpipe2/wembedding_service/compute_wembeddings.py \
    --format=conllu \
    --model "$WEMB_MODEL" \
    --mask_ellipsis \
    --ellipsis_mask_token "$MASK_TOKEN" \
    "$inpath" "$outpath"
}

WEMB_WORKERS=2
WEMB_PIDS=()

run_wemb_pool() {
  local file="$1"
  while (( ${#WEMB_PIDS[@]} >= WEMB_WORKERS )); do
    wait -n
    local alive=()
    for pid in "${WEMB_PIDS[@]}"; do
      kill -0 "$pid" 2>/dev/null && alive+=("$pid")
    done
    WEMB_PIDS=("${alive[@]}")
  done
  compute_wemb "$file" &
  WEMB_PIDS+=($!)
}

for corpus in "${CV_CORPORA[@]}"; do
  run_wemb_pool "datasets_mmbert/${corpus}/test.conllu"
done
for pid in "${WEMB_PIDS[@]}"; do wait "$pid"; done

echo "=== [STEP 2] Wembeddings done ==="

# 3. Install UDPipe2 deps into TF1 env
echo "=== [STEP 3] Installing UDPipe 2 runtime deps into TF1 env ==="

deactivate || true

pip install \
  "protobuf==3.20.3" \
  tqdm \
  "tensorboard==1.15" \
  ufal.udpipe \
  ufal.chu_liu_edmonds

echo "=== [STEP 3] UDPipe2 deps ready ==="

# 4. Predict: 12 (corpus, run) pairs in parallel
echo "=== [STEP 4] Running predictions ==="

NUM_RUNS=5

predict_fold() {
  local corpus="$1"
  local run_idx="$2"
  local model_dir="models/UDPipe2/${WEMB_MODEL_TAG}/run${run_idx}/${corpus}"
  local in_file="datasets_mmbert/${corpus}/test.conllu"
  local out_file="out/UDPipe2/${WEMB_MODEL_TAG}/run${run_idx}/${corpus}/test.pred.conllu"

  if [[ ! -d "$model_dir" ]]; then
    echo "[skip] model not found: ${model_dir}"
    return
  fi
  if [[ ! -f "$in_file" ]]; then
    echo "[skip] input not found: ${in_file}"
    return
  fi

  mkdir -p "$(dirname "$out_file")"
  echo "[pred] run${run_idx}/${corpus}"

  python vendor/udpipe2/udpipe2.py "$model_dir" \
    --predict \
    --predict_input "$in_file" \
    --predict_output "$out_file"

  if [[ -s "$out_file" ]]; then
    echo "[ok  ] ${out_file}"
  else
    echo "[warn] empty output: ${out_file}"
  fi
}

export -f predict_fold
export WEMB_MODEL_TAG

# Build list of all (corpus, run) pairs
ALL_PAIRS=()
for corpus in "${CV_CORPORA[@]}"; do
  for run_idx in $(seq 1 "$NUM_RUNS"); do
    ALL_PAIRS+=("${corpus}:${run_idx}")
  done
done

BATCH_SIZE=12
total=${#ALL_PAIRS[@]}
for (( i=0; i<total; i+=BATCH_SIZE )); do
  batch=("${ALL_PAIRS[@]:$i:$BATCH_SIZE}")
  echo "--- Batch $((i/BATCH_SIZE + 1)): ${batch[*]} ---"
  PIDS=()
  for pair in "${batch[@]}"; do
    corpus="${pair%%:*}"
    run_idx="${pair##*:}"
    predict_fold "$corpus" "$run_idx" &
    PIDS+=($!)
  done
  for pid in "${PIDS[@]}"; do wait "$pid"; done
  echo "--- Batch $((i/BATCH_SIZE + 1)) done ---"
done

echo "=== [ALL DONE] Predictions are in out/UDPipe2/${WEMB_MODEL_TAG}/ ==="
exec bash
