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

# 1. Create env for wembeddings using uv
echo "=== [STEP 1] Setting up uv venv for wembeddings ==="

if [[ -d ".venv-wemb-modern" ]]; then
  echo "[INFO] Existing .venv-wemb-modern found, reusing it"
else
  uv venv --python 3.12 .venv-wemb-modern
fi

source .venv-wemb-modern/bin/activate

uv pip install torch --index-url https://download.pytorch.org/whl/cu121
if [[ -f vendor/udpipe2/wembedding_service/requirements.txt ]]; then
  uv pip install -r vendor/udpipe2/wembedding_service/requirements.txt
else
  uv pip install numpy scipy tqdm
fi

echo "[STEP 1] venv ready."

# 2. Compute wembeddings for test sets (skip if .npz already exists)
echo "=== [STEP 2] Computing wembeddings for test sets (ModernBERT) ==="

WEMB_PY=".venv-wemb-modern/bin/python"
FORCE=0
WEMB_MODEL="modernbert-base-last4"
MASK_ELLIPSIS=1
MASK_TOKEN="[MASK]"

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
  "$WEMB_PY" vendor/udpipe2/wembedding_service/compute_wembeddings.py \
    --format=conllu \
    --model "$WEMB_MODEL" \
    --mask_ellipsis \
    --ellipsis_mask_token "$MASK_TOKEN" \
    "$inpath" "$outpath"
}

CORPORA=(ud ud-new ud-old str str-new str-old)

for corpus in "${CORPORA[@]}"; do
  compute_wemb "datasets_modern/${corpus}/test.conllu"
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

# 4. Predict with all trained ModernBERT models
echo "=== [STEP 4] Running predictions for all ModernBERT models ==="

WEMB_MODEL_TAG="ModernBERT"
SEEDS=(7 91 333 678 1999)
NUM_RUNS=${#SEEDS[@]}

predict_udpipe () {
  local corpus="$1"
  local run_idx="$2"
  local model_dir="models/UDPipe2/${WEMB_MODEL_TAG}/run${run_idx}/${corpus}"
  local in_file="datasets_modern/${corpus}/test.conllu"
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
  echo "[pred] corpus=${corpus} run=${run_idx}/${NUM_RUNS}"
  echo "       model=${model_dir}"
  echo "       input=${in_file}"
  echo "       output=${out_file}"

  python vendor/udpipe2/udpipe2.py "$model_dir" \
    --predict \
    --predict_input "$in_file" \
    --predict_output "$out_file"

  echo "[done] ${out_file}"
}

for corpus in "${CORPORA[@]}"; do
  for i in "${!SEEDS[@]}"; do
    run_idx=$((i + 1))
    predict_udpipe "$corpus" "$run_idx"
  done
done

echo "=== [ALL DONE] Predictions are in out/UDPipe2/${WEMB_MODEL_TAG}/ ==="
exec bash
