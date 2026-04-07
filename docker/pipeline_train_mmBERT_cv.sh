#!/bin/bash
set -euo pipefail

# Cross-validation training for mmBERT.
# 9 folds × 2 corpora = 18 (corpus, fold) pairs.

# 0. Ensure uv is installed and on PATH
if ! command -v uv >/dev/null 2>&1; then
  echo "[BOOT] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$PATH"

echo "[INFO] uv path: $(command -v uv || echo 'NOT FOUND')"
echo "[INFO] python default: $(command -v python)"

# 1. Create env for wembeddings
echo "=== [STEP 1] Creating uv venv for wembeddings ==="

rm -rf .venv-wemb-mmbert-cv/
uv venv --python 3.12 .venv-wemb-mmbert-cv
source .venv-wemb-mmbert-cv/bin/activate

uv pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu
if [[ -f vendor/udpipe2/wembedding_service/requirements.txt ]]; then
  uv pip install -r vendor/udpipe2/wembedding_service/requirements.txt
else
  uv pip install numpy scipy tqdm
fi
uv pip install "transformers>=4.48.0" safetensors

echo "[STEP 1] venv ready."

# 2. Compute wembeddings for all CV folds in parallel
echo "=== [STEP 2] Computing wembeddings ==="

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

export -f compute_wemb
export WEMB_PY WEMB_MODEL MASK_TOKEN

# All wembedding files in parallel
WEMB_PIDS=()
for corpus in "${CV_CORPORA[@]}"; do
  for split in train dev test; do
    compute_wemb "datasets_mmbert/${corpus}/${split}.conllu" &
    WEMB_PIDS+=($!)
  done
done

echo "[WEMB] Waiting for ${#WEMB_PIDS[@]} wembedding jobs..."
for pid in "${WEMB_PIDS[@]}"; do
  wait "$pid"
done
echo "=== [STEP 2] Wembeddings done ==="

# 3. Install UDPipe2 deps into TF1 env
echo "=== [STEP 3] Installing UDPipe 2 runtime deps ==="
deactivate || true
pip install \
  "protobuf==3.20.3" \
  tqdm \
  "tensorboard==1.15" \
  ufal.udpipe \
  ufal.chu_liu_edmonds
echo "=== [STEP 3] UDPipe2 deps ready ==="

# 4. Train: (corpus, fold) pairs in parallel, 5 seeds sequentially each
echo "=== [STEP 4] Training (3 pairs in parallel × 5 seeds) ==="

mkdir -p "models/UDPipe2/${WEMB_MODEL_TAG}"

SEEDS=(7 91 333 678 1999)

train_fold() {
  local corpus="$1"
  local train_file="datasets_mmbert/${corpus}/train.conllu"
  local dev_file="datasets_mmbert/${corpus}/dev.conllu"

  if [[ ! -f "$train_file" ]]; then
    echo "[skip] $train_file not found" >&2
    return 0
  fi

  local seeds=(7 91 333 678 1999)
  for i in "${!seeds[@]}"; do
    local run_idx=$((i + 1))
    local seed="${seeds[$i]}"
    local model_dir="models/UDPipe2/${WEMB_MODEL_TAG}/run${run_idx}/${corpus}"

    echo "[train] ${corpus} seed=${seed} run=${run_idx}/5"
    python vendor/udpipe2/udpipe2.py "$model_dir" \
      --train "$train_file" \
      --dev "$dev_file" \
      --seed "$seed" \
      --wembedding_model "$WEMB_MODEL" \
      --max_sentence_len 256 \
      --parse 1 \
      --tags "UPOS,FEATS" \
      --threads 8 \
      --mask_ellipsis \
      --ellipsis_mask_token "$MASK_TOKEN"
    echo "[done] ${corpus} run${run_idx}"
  done
}

export -f train_fold
export WEMB_MODEL MASK_TOKEN

# Build list of all (corpus, fold) pairs
ALL_PAIRS=()
for corpus in "${CV_CORPORA[@]}"; do
  ALL_PAIRS+=("$corpus")
done

# Run in batches in parallel
BATCH_SIZE=12
total=${#ALL_PAIRS[@]}
for (( i=0; i<total; i+=BATCH_SIZE )); do
  batch=("${ALL_PAIRS[@]:$i:$BATCH_SIZE}")
  echo "--- Batch $((i/BATCH_SIZE + 1)): ${batch[*]} ---"
  PIDS=()
  for corpus in "${batch[@]}"; do
    train_fold "$corpus" &
    PIDS+=($!)
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
  echo "--- Batch $((i/BATCH_SIZE + 1)) done ---"
done

echo "=== [ALL DONE] CV models in models/UDPipe2/${WEMB_MODEL_TAG}/ ==="
exec bash
