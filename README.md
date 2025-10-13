# SynTagRus-UD-Comparison
Comparison of two corpora with different markup (SynTagRus and Universal Dependencies).

[Paper link](https://www.elibrary.ru/item.asp?id=78752085)

## UDPipe 2 Morphology Training (with ready wembeddings)
### GPU (Docker + NGC Tensorflow 1 Container)

0) Download and setup docker

1) Login to NGC
```
docker login nvcr.io
# Username: $oauthtoken
# Password: API key 
```

2) Launch the TF1 Container with CUDA 11.x
```
docker run --gpus all -it --rm \
  -v "$PWD":/workspace -w /workspace \
  -p 6006:6006 \
  nvcr.io/nvidia/tensorflow:22.12-tf1-py3 \
  bash -c "bash docker/setup_udpipe2_env.sh && bash"
```

1) Train UDPipe 2 on UD-SynTagRus
```
python vendor/udpipe2/udpipe2.py models/ud-morph \
  --train datasets/ud/train.conllu \
  --dev datasets/ud/dev.conllu \
  --max_sentence_len 256 \
  --parse 0 \
  --tags "UPOS,FEATS" \
  --threads 8
```

1) Train UDPipe 2 on SynTagRus
```
python vendor/udpipe2/udpipe2.py models/str-morph \
  --train datasets/str/train.conllu \
  --dev datasets/str/dev.conllu \
  --max_sentence_len 256 \
  --parse 0 \
  --tags "UPOS,FEATS" \
  --threads 8
```

## UDPipe 2 Morphology Evaluating

0) Create and activate environment with dependencies for inference
```
pip install uv      # uv is used for example
uv venv .venv-udpipe
source .venv-udpipe/bin/activate
uv pip install "tensorflow>=2.3.1" "transformers>=4,<5" ufal.chu_liu_edmonds ufal.udpipe
```

1) Launch UDPipe 2 server with morphology models loaded
```
python vendor/udpipe2/udpipe2_server.py 8001 ud-morph \
ud-morph models/ud-morph/ ru_syntagrus None \
str-morph models/str-morph/ ru_syntagrus None
```

2) Compute predictions by chunks script on UD-SynTagRus
```
python scripts/predict_conllu_in_chunks.py \
  --input datasets/ud/test.conllu \
  --service http://localhost:8001 \
  --model ud-morph \
  --chunk_size 2000 \
  --chunks_dir tmp \
  --chunks_pred_dir out/ud-morph/chunks \
  --client_script vendor/udpipe2/udpipe2_client.py \
  --output out/ud-morph/test.pred.conllu \
  --tagger udpipe
```

3) Compute predictions by chunks script on SynTagRus
```
python scripts/predict_conllu_in_chunks.py \
  --input datasets/str/test.conllu \
  --service http://localhost:8001 \
  --model str-morph \
  --chunk_size 2000 \
  --chunks_dir tmp \
  --chunks_pred_dir out/str-morph/chunks \
  --client_script vendor/udpipe2/udpipe2_client.py \
  --output out/str-morph/test.pred.conllu \
  --tagger udpipe
```

4) Evaluate both corpora predictions
```
python scripts/evaluate_morph.py datasets/ud/test.conllu out/ud-morph/test.pred.conllu

python scripts/evaluate_morph.py datasets/str/test.conllu out/str-morph/test.pred.conllu
```