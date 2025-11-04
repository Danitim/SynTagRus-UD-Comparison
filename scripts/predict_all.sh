set -euo pipefail

CORPORA=(ud ud-new ud-old str str-new str-old)
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

echo "=== [PREDICT] Running predictions ==="
for corpus in "${CORPORA[@]}"; do
  model_dir="models/${corpus}-morph"
  in_file="datasets/${corpus}/test.conllu"
  out_dir="out/${corpus}"
  out_file="${out_dir}/test.pred.conllu"

  if [[ ! -d "$model_dir" ]]; then
    echo "[SKIP] Model not found: ${model_dir}"
    continue
  fi
  if [[ ! -f "$in_file" ]]; then
    echo "[SKIP] Input test file not found: ${in_file}"
    continue
  fi

  mkdir -p "$out_dir"
  echo "[RUN ] ${corpus}: ${in_file} -> ${out_file}"

  $PYTHON "$UDPIPE2" "$model_dir" \
    --predict \
    --predict_input "$in_file" \
    --predict_output "$out_file"

  if [[ -s "$out_file" ]]; then
    echo "[OK  ] ${corpus} -> ${out_file}"
  else
    echo "[WARN] Empty output for ${corpus}: ${out_file}"
  fi
done

echo "=== [DONE] Done. Predictions are in out/<corpus>/test.pred.conllu ==="
