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
  nvcr.io/nvidia/tensorflow:22.12-tf1-py3 bash
```

3) Install UDPipe 2 dependencies inside the container:
```
python -m pip install --no-cache-dir \
  "protobuf==3.20.3" tqdm "tensorboard==1.15" ufal.chu_liu_edmonds
```

4) Train UDPipe 2 on UD corpus
```
python vendor/udpipe2/udpipe2.py models/ud-morph \
  --train datasets/ud/train.conllu \
  --dev datasets/ud/dev.conllu \
  --max_sentence_len 256 \
  --parse 0 \
  --tags "UPOS,FEATS" \
  --threads 8
```

5) Train UDPipe 2 on SynTagRus corpus
```
python vendor/udpipe2/udpipe2.py models/str-morph \
  --train datasets/str/train.conllu \
  --dev datasets/str/dev.conllu \
  --max_sentence_len 256 \
  --parse 0 \
  --tags "UPOS,FEATS" \
  --threads 8
```

