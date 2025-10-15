import argparse
import io
import os
import sys
from typing import List, Tuple

ADDED_PREFIX = "# Added to SynTagRus"

def read_sentences(path: str) -> List[List[str]]:
    sentences: List[List[str]] = []
    cur: List[str] = []
    with io.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "":
                if cur:
                    sentences.append(cur)
                    cur = []
            else:
                cur.append(line)
        if cur:
            sentences.append(cur)
    return sentences

def is_ud_sentence_new(ud_sent_lines: List[str]) -> bool:
    for ln in ud_sent_lines:
        if ln.lstrip().startswith("#") and ln.lstrip().startswith(ADDED_PREFIX):
            return True
    return False

def write_sentences(path: str, sents: List[List[str]]) -> None:
    with io.open(path, 'w', encoding='utf-8', newline='') as f:
        for i, sent in enumerate(sents):
            for ln in sent:
                f.write(ln if ln.endswith("\n") else ln + "\n")
            if i != len(sents) - 1:
                f.write("\n")

def split_aligned(
    ud_path: str,
    str_path: str,
    outdir: str,
) -> Tuple[int, int, int]:
    ud_sents = read_sentences(ud_path)
    str_sents = read_sentences(str_path)

    if len(ud_sents) != len(str_sents):
        msg = f"Предупреждение: число предложений не совпадает (UD={len(ud_sents)}, STR={len(str_sents)})."
        print(msg, file=sys.stderr)
        sys.exit(2)

    n = min(len(ud_sents), len(str_sents))

    ud_new, ud_old, str_new, str_old = [], [], [], []
    n_new = 0
    for i in range(n):
        if is_ud_sentence_new(ud_sents[i]):
            ud_new.append(ud_sents[i]); str_new.append(str_sents[i]); n_new += 1
        else:
            ud_old.append(ud_sents[i]); str_old.append(str_sents[i])

    os.makedirs(outdir, exist_ok=True)
    write_sentences(os.path.join(outdir, "ud_new.conllu"),  ud_new)
    write_sentences(os.path.join(outdir, "ud_old.conllu"),  ud_old)
    write_sentences(os.path.join(outdir, "str_new.conllu"), str_new)
    write_sentences(os.path.join(outdir, "str_old.conllu"), str_old)

    print(f"[OK] Записано:\n"
          f"  UD:  new={len(ud_new)} → {os.path.join(outdir, 'ud_new.conllu')}\n"
          f"       old={len(ud_old)} → {os.path.join(outdir, 'ud_old.conllu')}\n"
          f"  STR: new={len(str_new)} → {os.path.join(outdir, 'str_new.conllu')}\n"
          f"       old={len(str_old)} → {os.path.join(outdir, 'str_old.conllu')}")
    return n, n_new, n - n_new

def main():
    ap = argparse.ArgumentParser(
        description="Разделение выровненных UD/STR .conllu на старые/новые по строке '# Added to SynTagRus' в UD."
    )
    ap.add_argument("--ud", required=True, help="Путь к ud_aligned.conllu")
    ap.add_argument("--str", required=True, help="Путь к str_aligned.conllu")
    ap.add_argument("--outdir", default=".", help="Каталог для вывода (по умолчанию текущий)")
    args = ap.parse_args()

    total, n_new, n_old = split_aligned(args.ud, args.str, args.outdir)
    print(f"Итог: использовано предложений: {total}; новых: {n_new}; старых: {n_old}")

if __name__ == "__main__":
    main()
