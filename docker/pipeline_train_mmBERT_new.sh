#!/bin/bash
set -euo pipefail

# Corpora: ud-new, str-new

# 0. Ensure uv is installed and on PATH
if ! command -v uv >/dev/null 2>&1; then
  echo "[BOOT] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$PATH"

echo "[INFO] uv path: $(command -v uv || echo 'NOT FOUND')"
echo "[INFO] python default: $(command -v python)"

# 1. Create env for wembeddings using uv
echo "=== [STEP 1] Creating uv venv for wembeddings ==="

rm -rf .venv-wemb-mmbert-new/
uv venv --python 3.12 .venv-wemb-mmbert-new

source .venv-wemb-mmbert-new/bin/activate

echo "=== [STEP 1] Installing wembeddings deps via uv ==="
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
if [[ -f vendor/udpipe2/wembedding_service/requirements.txt ]]; then
  uv pip install -r vendor/udpipe2/wembedding_service/requirements.txt
else
  echo "[WARN] requirements.txt for wembedding_service not found, installing minimal deps"
  uv pip install numpy scipy tqdm
fi

echo "[STEP 1] venv ready."
echo "[INFO] Python in wemb venv: $(which python)"

# 2. Compute wembeddings for all datasets
echo "=== [STEP 2] Computing wembeddings for corpora: ud-new, str-new ==="

WEMB_PY=".venv-wemb-mmbert-new/bin/python"
FORCE=1
MASK_ELLIPSIS=1
MASK_TOKEN="[MASK]"
WEMB_MODEL="mmBERT-base-last4"
WEMB_MODEL_TAG="mmBERT"

compute_wemb () {
  local inpath="$1"
  local outpath="${inpath}.npz"

  if [[ ! -f "$inpath" ]]; then
    echo "[warn] $inpath not found, skipping" >&2
    return 0
  fi

  if [[ -f "$outpath" && "$FORCE" != "1" ]]; then
    echo "[skip] already have $outpath"
    return 0
  fi

  echo "[WEMB] $inpath -> $outpath"
  if [[ "$MASK_ELLIPSIS" == "1" ]]; then
    "$WEMB_PY" vendor/udpipe2/wembedding_service/compute_wembeddings.py \
      --format=conllu \
      --model "$WEMB_MODEL" \
      --mask_ellipsis \
      --ellipsis_mask_token "$MASK_TOKEN" \
      "$inpath" "$outpath"
  else
    "$WEMB_PY" vendor/udpipe2/wembedding_service/compute_wembeddings.py \
      --format=conllu \
      --model "$WEMB_MODEL" \
      "$inpath" "$outpath"
  fi
}

CORPORA=(ud-new str-new)

for corpus in "${CORPORA[@]}"; do
  for split in train dev test; do
    compute_wemb "datasets/${corpus}/${split}.conllu"
  done
done

echo "=== [STEP 2] Wembeddings done ==="

# 3. Install UDPipe2 deps into container's main Python (TF1 env)
echo "=== [STEP 3] Installing UDPipe 2 runtime deps into TF1 env ==="

deactivate || true

pip install \
  "protobuf==3.20.3" \
  tqdm \
  "tensorboard==1.15" \
  ufal.udpipe \
  ufal.chu_liu_edmonds

echo "=== [STEP 3] UDPipe2 deps ready ==="

# 4. Train 5 UDPipe 2 models with different random seeds
echo "=== [STEP 4] Training UDPipe 2 models (5 runs × ${#CORPORA[@]} corpora) ==="
mkdir -p "models/UDPipe2/"
mkdir -p "models/UDPipe2/${WEMB_MODEL_TAG}"

SEEDS=(7 91 333 678 1999)
NUM_SEEDS=${#SEEDS[@]}

train_udpipe () {
  local corpus="$1"
  local seed="$2"
  local run_idx="$3"
  local model_dir="models/UDPipe2/${WEMB_MODEL_TAG}/run${run_idx}/${corpus}"
  local train_file="datasets/${corpus}/train.conllu"
  local dev_file="datasets/${corpus}/dev.conllu"

  if [[ ! -f "$train_file" ]]; then
    echo "[skip] $train_file not found, skipping ${corpus}"
    return
  fi

  echo "[train] corpus=${corpus} seed=${seed} run=${run_idx}/${NUM_SEEDS}"
  echo "        train=${train_file}"
  echo "        dev=${dev_file}"
  echo "        model_dir=${model_dir}"

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
}

for corpus in "${CORPORA[@]}"; do
  for i in "${!SEEDS[@]}"; do
    run_idx=$((i + 1))
    mkdir -p "models/UDPipe2/${WEMB_MODEL_TAG}/run${run_idx}"
    train_udpipe "$corpus" "${SEEDS[$i]}" "$run_idx"
  done
done

echo "=== [ALL DONE] Corpora: ud-new, str-new — models in models/UDPipe2/${WEMB_MODEL_TAG}/ ==="
exec bash
