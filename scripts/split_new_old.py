import argparse
from pathlib import Path
import pyconll
from typing import List, Tuple

NEW_MARKER = "Added to SynTagRus"

def write_sentences(path: Path, sentences: List[pyconll.unit.sentence.Sentence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        for i, sent in enumerate(sentences):
            f.write(sent.conll())
            f.write("\n")
            if i != len(sentences) - 1:
                f.write("\n")

def sentence_is_new(ud_sent: pyconll.unit.sentence.Sentence) -> bool:
    for line in ud_sent.conll().splitlines():
        if f"# {NEW_MARKER}" in line.strip():
            return True
    return False

def split_pair(
    ud_in: Path,
    str_in: Path,
) -> Tuple[List[pyconll.unit.sentence.Sentence],
           List[pyconll.unit.sentence.Sentence],
           List[pyconll.unit.sentence.Sentence],
           List[pyconll.unit.sentence.Sentence]]:
    ud_conll = pyconll.load_from_file(ud_in)
    str_conll = pyconll.load_from_file(str_in)

    if len(ud_conll) != len(str_conll):
        raise RuntimeError(
            f"Split mismatch: {ud_in} has {len(ud_conll)} sentences, "
            f"but {str_in} has {len(str_conll)} sentences"
        )

    ud_old, ud_new = [], []
    str_old, str_new = [], []

    for u_sent, s_sent in zip(ud_conll, str_conll):
        if sentence_is_new(u_sent):
            ud_new.append(u_sent)
            str_new.append(s_sent)
        else:
            ud_old.append(u_sent)
            str_old.append(s_sent)

    return ud_old, ud_new, str_old, str_new

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Берёт datasets/ud/{train,dev,test}.conllu и datasets/str/{train,dev,test}.conllu, "
            "делит каждую пару на old/new по маркеру '# Added to SynTagRus 2015–2020', "
            "и пишет в datasets/{ud-old,ud-new,str-old,str-new}/{train,dev,test}.conllu"
        )
    )
    ap.add_argument("--datasets-root", default="datasets",
                    help="Корень с поддиректориями ud/ и str/ (по умолчанию: datasets)")
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"],
                    help="Какие сплиты обрабатывать (по умолчанию: train dev test)")
    args = ap.parse_args()

    root = Path(args.datasets_root)

    for split_name in args.splits:
        ud_in = root / "ud" / f"{split_name.conllu if split_name.endswith('.conllu') else split_name + '.conllu'}"
        str_in = root / "str" / f"{split_name.conllu if split_name.endswith('.conllu') else split_name + '.conllu'}"

        # Вдруг кто-то вызовет со split_name='train.conllu' — поддержим оба случая.
        if not ud_in.exists():
            ud_in = root / "ud" / f"{split_name}.conllu"
        if not str_in.exists():
            str_in = root / "str" / f"{split_name}.conllu"

        ud_old_sents, ud_new_sents, str_old_sents, str_new_sents = split_pair(ud_in, str_in)

        # Куда писать
        out_ud_old = root / "ud-old" / f"{split_name}.conllu"
        out_ud_new = root / "ud-new" / f"{split_name}.conllu"
        out_str_old = root / "str-old" / f"{split_name}.conllu"
        out_str_new = root / "str-new" / f"{split_name}.conllu"

        write_sentences(out_ud_old, ud_old_sents)
        write_sentences(out_ud_new, ud_new_sents)
        write_sentences(out_str_old, str_old_sents)
        write_sentences(out_str_new, str_new_sents)

        print(
            f"[{split_name}] "
            f"ud_old={len(ud_old_sents)} ud_new={len(ud_new_sents)} | "
            f"str_old={len(str_old_sents)} str_new={len(str_new_sents)}"
        )

if __name__ == "__main__":
    main()
