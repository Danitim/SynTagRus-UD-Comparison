# SynTagRus-UD-Comparison
Comparison of two corpora with different markup (SynTagRus and Universal Dependencies).

[Paper link](https://www.elibrary.ru/item.asp?id=78752085)

## Dataset preparation
0) Download SynTagRus and SynTagRus UD version into `Corpora` folder (preferably with names `SynTagRus` and `UD_SynTagRus`)

1) Install required libraries (e.g. ```pip install -r convert/requirements.txt```)

2) Convert corpora to the same format:
```
python3 corpora.py convert --corpus UD
python3 corpora.py convert --corpus STR
```

3) Align both corpora:
```
python3 corpora.py align
```

4) Restore the initial splits (used in UD version):
```
python3 scripts/restore_splits.py
```

5) Create splits for new/old data:
```
python3 scripts/split_new_old.py 
```

6) Fix possible last blank line errors:
```
./scripts/fix_conllu_eof.sh
```

## UDPipe 2 Morphology Training (with ready wembeddings)
### GPU (Docker + NGC Tensorflow 1 Container)

0) Download, setup and launch docker

1) Login to NGC
```
docker login nvcr.io
# Username: $oauthtoken
# Password: API key 
```

2) Launch the training process:
```
docker run --gpus all -it --rm \
  --ipc=host --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$PWD":/workspace -w /workspace \
  -p 6006:6006 \
  nvcr.io/nvidia/tensorflow:22.12-tf1-py3 \
  bash docker/pipeline_all.sh
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