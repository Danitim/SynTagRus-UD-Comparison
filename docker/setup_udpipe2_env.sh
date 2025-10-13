#!/bin/bash
set -e

echo "[INFO] Checking TensorFlow and GPU..."
python - <<'PY'
import tensorflow as tf
print("TF version:", tf.__version__)
print("GPU available:", tf.test.is_gpu_available(cuda_only=True))
PY

echo "[INFO] Installing UDPipe 2 dependencies..."
python -m pip install --no-cache-dir \
  "protobuf==3.20.3" tqdm "tensorboard==1.15" ufal.chu_liu_edmonds

echo "[INFO] Environment ready."
