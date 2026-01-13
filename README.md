# SynTagRus-UD-Comparison
Comparison of two corpora with different markup (SynTagRus and Universal Dependencies).

[Paper link](https://itas2024.iitp.ru/media/upload/itas-2024.pdf#page=330)

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

1) Login to NGC:
```
docker login nvcr.io
# Username: $oauthtoken
# Password: API key 
```

2) Launch the training process:
```
docker run --gpus '"device=0"' -it --rm \
  --ipc=host --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$PWD":/workspace -w /workspace \
  -p 127.0.0.1:6006:6006 \
  nvcr.io/nvidia/tensorflow:22.12-tf1-py3 \
  bash docker/pipeline_tag.sh
```

## UDPipe 2 Morphology Evaluating

1) Run prediction compute:
```
./scripts/predict_morph.sh
```

2) Evaluate morphology and write results:
```
python3 scripts/evaluate_morph.py
```
