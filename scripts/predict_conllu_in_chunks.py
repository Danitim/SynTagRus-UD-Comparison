# scripts/predict_conllu_in_chunks.py
#!/usr/bin/env python3
import argparse
import pathlib
import subprocess
import sys
import shutil

def split_conllu(in_path: pathlib.Path, out_dir: pathlib.Path, n_per_chunk: int):
    """Делит CoNLL-U по предложениям (пустая строка = граница). Возвращает список путей и списки диапазонов предложений."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parts, ranges = [], []
    buf = []
    sent = 0
    start_sent = 0
    k = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            buf.append(line)
            if line.strip() == "":
                sent += 1
                if sent % n_per_chunk == 0:
                    k += 1
                    part = out_dir / f"test.part{k:04d}.conllu"
                    part.write_text("".join(buf), encoding="utf-8")
                    parts.append(part)
                    ranges.append((start_sent, sent))
                    buf = []
                    start_sent = sent
        if buf:
            k += 1
            part = out_dir / f"test.part{k:04d}.conllu"
            part.write_text("".join(buf), encoding="utf-8")
            parts.append(part)
            ranges.append((start_sent, sent))

    if not parts:
        raise SystemExit("Не создался ни один чанк — проверьте входной файл.")
    return parts, ranges, sent

def run_client_one(client: pathlib.Path, service: str, model: str,
                   out_dir: pathlib.Path, chunk_conllu: pathlib.Path,
                   tagger="none", parser="none"):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (chunk_conllu.stem.replace(".conllu", "") + ".pred.conllu")  # test.part0001.pred.conllu

    cmd = [
        sys.executable, str(client),
        "--service", service,
        "--model", model,
        "--input", "conllu",
        "--output", "conllu",
        "--outfile", str(out_path)
    ]
    if tagger and tagger != "none":
        cmd = cmd + ["--tagger", tagger]
    if parser and parser != "none":
        cmd = cmd + ["--parser", parser]
    cmd.append(str(chunk_conllu))

    print(">>", chunk_conllu.name, ":", " ".join(cmd))
    r = subprocess.run(cmd, text=True, capture_output=True)
    if r.returncode != 0:
        log = out_dir / (chunk_conllu.name + ".stderr.log")
        log.write_text(f"RC={r.returncode}\nSTDOUT:\n{r.stdout}\n\nSTDERR:\n{r.stderr}\n", encoding="utf-8")
        raise RuntimeError(f"{chunk_conllu.name} FAIL — см. {log}")


def concat_predictions(chunks_pred_dir: pathlib.Path, out_file: pathlib.Path):
    preds = sorted(chunks_pred_dir.glob("test.part*.pred.conllu"))
    if not preds:
        preds = sorted(chunks_pred_dir.glob("test.part*.conllu.pred.conllu"))
    if not preds:
        raise SystemExit(f"Не найдены предсказания в {chunks_pred_dir}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as w:
        for p in preds:
            w.write(p.read_text(encoding="utf-8"))
    print(f">> Склеено {len(preds)} частей → {out_file}")


def main():
    ap = argparse.ArgumentParser(description="Нарезать CoNLL-U под чанки, предсказать UDPipe2-клиентом и склеить.")
    ap.add_argument("--input", default="datasets/ud/test.conllu")
    ap.add_argument("--service", default="http://localhost:8001")
    ap.add_argument("--model", default="ud-morph")
    ap.add_argument("--chunk_size", type=int, default=2000)
    ap.add_argument("--chunks_dir", default="tmp/ud_chunks")
    ap.add_argument("--chunks_pred_dir", default="out/ud-morph/chunks")
    ap.add_argument("--client_script", default="vendor/udpipe2/udpipe2_client.py")
    ap.add_argument("--output", default="out/ud-morph/test.pred.conllu")
    ap.add_argument("--tagger", default="none", choices=["none","udpipe"])
    ap.add_argument("--parser", default="none", choices=["none","udpipe"])
    args = ap.parse_args()

    in_path = pathlib.Path(args.input)

    # 1) Разбивка CoNLL-U
    chunk_dir = pathlib.Path(args.chunks_dir)
    print(f">>> Разбиваем {in_path} на чанки по {args.chunk_size} предложений…")
    parts, ranges, total = split_conllu(in_path, chunk_dir, args.chunk_size)
    print(f">>> Чанков: {len(parts)}; предложений всего: {total}")

    # 2) Предсказания по чанкам
    pred_dir = pathlib.Path(args.chunks_pred_dir)
    for p in parts:
        run_client_one(
            client=pathlib.Path(args.client_script),
            service=args.service,
            model=args.model,
            out_dir=pred_dir,
            chunk_conllu=p,
            tagger=args.tagger,
            parser=args.parser,
        )

    # 3) Склейка и очистка tmp и chunks
    concat_predictions(pred_dir, pathlib.Path(args.output))
    shutil.rmtree(chunk_dir, ignore_errors=True)
    shutil.rmtree(args.chunks_pred_dir, ignore_errors=True)
    print(f">> Удалили временную папку {chunk_dir}")

if __name__ == "__main__":
    main()
