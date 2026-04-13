#!/bin/bash
set -euo pipefail

# Cross-validation training for mmBERT — corpora: str-old, ud-old, str (full), ud (full).
# 9 folds × 4 corpora = 36 (corpus, fold) pairs.
# GPU budget: ≤32 GB  →  BATCH_SIZE=6 (6 × 4.6 GB ≈ 27.6 GB).
# wembeddings: WEMB_WORKERS=2 (server limit).

# 0. Ensure uv is installed and on PATH
if ! command -v uv >/dev/null 2>&1; then
  echo "[BOOT] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$PATH"

echo "[INFO] uv path: $(command -v uv || echo 'NOT FOUND')"
echo "[INFO] python default: $(command -v python)"

# 1. Create / reuse wembeddings venv
echo "=== [STEP 1] Setting up uv venv for wembeddings ==="

if [[ -d ".venv-wemb-mmbert-cv" ]]; then
  echo "[INFO] Reusing existing .venv-wemb-mmbert-cv"
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

# 2. Compute wembeddings for all CV folds (max 2 in parallel)
echo "=== [STEP 2] Computing wembeddings ==="

WEMB_PY=".venv-wemb-mmbert-cv/bin/python"
WEMB_MODEL="mmBERT-base-last4"
WEMB_MODEL_TAG="mmBERT"
MASK_TOKEN="<mask>"

CV_CORPORA=(
  str-old-cv1 str-old-cv2 str-old-cv3 str-old-cv4
  str-old-cv5 str-old-cv6 str-old-cv7 str-old-cv8 str-old-cv9
  ud-old-cv1  ud-old-cv2  ud-old-cv3  ud-old-cv4
  ud-old-cv5  ud-old-cv6  ud-old-cv7  ud-old-cv8  ud-old-cv9
  str-cv1 str-cv2 str-cv3 str-cv4
  str-cv5 str-cv6 str-cv7 str-cv8 str-cv9
  ud-cv1  ud-cv2  ud-cv3  ud-cv4
  ud-cv5  ud-cv6  ud-cv7  ud-cv8  ud-cv9
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
  for split in train dev test; do
    run_wemb_pool "datasets_mmbert/${corpus}/${split}.conllu"
  done
done

for pid in "${WEMB_PIDS[@]}"; do wait "$pid"; done
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

# 4. Train: BATCH_SIZE=6 → 6 × 4.6 GB ≈ 27.6 GB VRAM (≤32 GB budget)
echo "=== [STEP 4] Training (6 pairs in parallel × 5 seeds sequential) ==="

mkdir -p "models/UDPipe2/${WEMB_MODEL_TAG}"

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
export WEMB_MODEL MASK_TOKEN WEMB_MODEL_TAG

BATCH_SIZE=6   # 6 × 4.6 GB = 27.6 GB — fits within 32 GB budget
total=${#CV_CORPORA[@]}
for (( i=0; i<total; i+=BATCH_SIZE )); do
  batch=("${CV_CORPORA[@]:$i:$BATCH_SIZE}")
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
