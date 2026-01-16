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
echo "=== [STEP 1] Creating uv venv for wembeddings ==="

if [[ -d ".venv-wemb" ]]; then
  echo "[INFO] Existing .venv-wemb found, reusing it"
else
  echo "[INFO] No .venv-wemb found, creating new one"
  uv venv .venv-wemb
fi

source .venv-wemb/bin/activate

echo "=== [STEP 1] Installing wembeddings deps via uv ==="
if [[ -f vendor/udpipe2/wembedding_service/requirements.txt ]]; then
  uv pip install -r vendor/udpipe2/wembedding_service/requirements.txt
else
  echo "[WARN] requirements.txt for wembedding_service not found, installing minimal deps"
  uv pip install numpy scipy tqdm
fi

echo "[STEP 1] venv ready."
echo "[INFO] Python in wemb venv: $(which python)"

# 2. Compute wembeddings for all datasets
echo "=== [STEP 2] Computing wembeddings for all corpora ==="

WEMB_PY=".venv-wemb/bin/python"
FORCE=0
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
  if [[ "$MASK_ELLIPSIS" == "1" ]]; then
    "$WEMB_PY" vendor/udpipe2/wembedding_service/compute_wembeddings.py \
      --format=conllu \
      --mask_ellipsis \
      --ellipsis_mask_token "$MASK_TOKEN" \
      "$inpath" "$outpath"
  else
    "$WEMB_PY" vendor/udpipe2/wembedding_service/compute_wembeddings.py \
      --format=conllu \
      "$inpath" "$outpath"
  fi
}

CORPORA=(ud ud-new ud-old str str-new str-old)

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

# 4. Train UDPipe 2 parsing-only models for all corpora
echo "=== [STEP 4] Training UDPipe 2 parsing-only models ==="
mkdir -p models

train_udpipe () {
  local corpus="$1"
  local model_dir="models/${corpus}-syntax"
  local train_file="datasets/${corpus}/train.conllu"
  local dev_file="datasets/${corpus}/dev.conllu"

  if [[ ! -f "$train_file" ]]; then
    echo "[skip] $train_file not found, skipping ${corpus}"
    return
  fi

  echo "[train] corpus=${corpus}"
  echo "        train=${train_file}"
  echo "        dev=${dev_file}"
  echo "        model_dir=${model_dir}"

  python vendor/udpipe2/udpipe2.py "$model_dir" \
    --train "$train_file" \
    --dev "$dev_file" \
    --max_sentence_len 256 \
    --parse 1 \
    --tags "" \
    --threads 8 \
    --mask_ellipsis \
    --ellipsis_mask_token "$MASK_TOKEN"
}

for corpus in "${CORPORA[@]}"; do
  train_udpipe "$corpus"
done

echo "=== [ALL DONE] All parsing-only models trained successfully ==="
exec bash
