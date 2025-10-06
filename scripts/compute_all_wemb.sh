set -euo pipefail

WEMB_PY="${WEMB_PY:-.venv-wemb/bin/python}"
FORCE="${FORCE:-1}"

echo "Using python: ${WEMB_PY}"
echo "Force recompute: ${FORCE}"

compute() {
  local in="$1"
  local out="${in}.npz"

  if [[ -f "$out" && "$FORCE" != "1" ]]; then
    echo "[skip] $out already exists"
    return 0
  fi

  echo "[run ] $in  ->  $out"
  "$WEMB_PY" vendor/wembedding_service/compute_wembeddings.py \
    --format=conllu \
    "$in" "$out"
}

for sub in str ud; do
  for split in train dev test; do
    f="datasets/${sub}/${split}.conllu"
    if [[ ! -f "$f" ]]; then
      echo "[warn] not found: $f" >&2
      continue
    fi
    compute "$f"
  done
done

echo "Done."
