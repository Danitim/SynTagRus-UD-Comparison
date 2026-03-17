#!/bin/bash
set -euo pipefail

# 0. Install basic tools & uv
apt-get update -qq && apt-get install -y -qq curl > /dev/null 2>&1

if ! command -v uv >/dev/null 2>&1; then
  echo "[BOOT] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$PATH"

echo "[INFO] uv path: $(command -v uv || echo 'NOT FOUND')"
echo "[INFO] python default: $(command -v python)"

# ── Configuration ────────────────────────────────────────────
ENCODER_MODEL="bert-base-multilingual-uncased"
MASK_TOKEN="[MASK]"
MODEL_TAG="mBERT"

CORPORA=(ud ud-new ud-old str str-new str-old)
SEEDS=(7 91 333 678 1999)
NUM_SEEDS=${#SEEDS[@]}

COLUMNS="upos feats head deprel"

# ── Step 1: Create venv & install deps ───────────────────────
echo "=== [STEP 1] Creating uv venv for CoBaLD ==="

rm -rf .venv-cobald
uv venv --python 3.12 .venv-cobald

source .venv-cobald/bin/activate

echo "=== [STEP 1] Installing CoBaLD deps via uv ==="
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install -e vendor/cobald

echo "[STEP 1] venv ready."
echo "[INFO] Python: $(which python)"
echo "[INFO] PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ── Step 2: Generate model config ────────────────────────────
echo "=== [STEP 2] Generating model config ==="

MODEL_CONFIG="models/CoBaLD/${MODEL_TAG}/model_config.json"
mkdir -p "models/CoBaLD/${MODEL_TAG}"

cat > "$MODEL_CONFIG" <<EOF
{
  "encoder_model_name": "${ENCODER_MODEL}",
  "ellipsis_mask_token": "${MASK_TOKEN}",
  "morphology_classifier_hidden_size": 512,
  "dependency_classifier_hidden_size": 128
}
EOF

echo "[INFO] Model config written to ${MODEL_CONFIG}"

# ── Step 3: Train models ─────────────────────────────────────
echo "=== [STEP 3] Training CoBaLD models (${NUM_SEEDS} runs per corpus) ==="

train_cobald () {
  local corpus="$1"
  local seed="$2"
  local run_idx="$3"
  local model_dir="models/CoBaLD/${MODEL_TAG}/${corpus}-run${run_idx}"
  local data_dir="datasets/${corpus}"
  local train_file="${data_dir}/train.conllu"

  if [[ ! -f "$train_file" ]]; then
    echo "[skip] $train_file not found, skipping ${corpus}"
    return
  fi

  echo "[train] corpus=${corpus} seed=${seed} run=${run_idx}/${NUM_SEEDS}"
  echo "        data_dir=${data_dir}"
  echo "        model_dir=${model_dir}"

  python vendor/cobald/train.py \
    --model_config "$MODEL_CONFIG" \
    --data_dir "$data_dir" \
    --columns $COLUMNS \
    --output_dir "$model_dir" \
    --seed "$seed"
}

for corpus in "${CORPORA[@]}"; do
  for i in "${!SEEDS[@]}"; do
    run_idx=$((i + 1))
    train_cobald "$corpus" "${SEEDS[$i]}" "$run_idx"
  done
done

echo "=== [ALL DONE] All CoBaLD models (${NUM_SEEDS} seeds × ${#CORPORA[@]} corpora) trained successfully ==="
exec bash
