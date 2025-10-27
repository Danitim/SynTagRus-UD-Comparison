import argparse
from pathlib import Path
import pyconll
from typing import Dict, List

def write_sentences(path: Path, sentences: List[pyconll.unit.sentence.Sentence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        for i, sent in enumerate(sentences):
            f.write(sent.conll())
            f.write("\n")
            if i != len(sentences) - 1:
                f.write("\n")

def restore_splits(
    aligned_path: Path,
    outdir: Path,
    prefix: str,
    keep_unknown: bool = True,
) -> None:
    conll = pyconll.load_from_file(aligned_path)

    buckets: Dict[str, List[pyconll.unit.sentence.Sentence]] = {
        "train": [],
        "dev": [],
        "test": [],
        "unknown": [],
    }

    for sent in conll:
        split_val = sent.meta_value("split")
        if split_val is None:
            split_val = "unknown"
        split_val = split_val.strip().lower()

        if split_val not in buckets:
            buckets["unknown"].append(sent)
        else:
            buckets[split_val].append(sent)

    base = outdir / prefix
    write_sentences(base / "train.conllu", buckets["train"])
    write_sentences(base / "dev.conllu",   buckets["dev"])
    write_sentences(base / "test.conllu",  buckets["test"])

    if keep_unknown and buckets["unknown"]:
        write_sentences(base / "unknown.conllu", buckets["unknown"])

    total = sum(len(v) for v in buckets.values())
    print(f"[{prefix}] {aligned_path}: total={total} "
          f"train={len(buckets['train'])} dev={len(buckets['dev'])} "
          f"test={len(buckets['test'])} unknown={len(buckets['unknown'])}")

def main():
    ap = argparse.ArgumentParser(
        description="Восстанавливает исходные train/dev/test сплиты по полю split в aligned-файлах."
    )
    ap.add_argument("--ud-aligned", default="Aligned/ud_aligned.conllu",
                    help="Путь к выравненому UD файлу (ud_aligned.conllu)")
    ap.add_argument("--str-aligned", default="Aligned/str_aligned.conllu",
                    help="Путь к выравненому SynTagRus файлу (str_aligned.conllu)")
    ap.add_argument("--outdir", default="datasets",
                    help="Корень, куда писать (datasets)")
    ap.add_argument("--no-unknown", action="store_true",
                    help="Не сохранять предложения с split=unknown")
    args = ap.parse_args()

    outdir = Path(args.outdir)

    restore_splits(
        aligned_path=Path(args.ud_aligned),
        outdir=outdir,
        prefix="ud",
        keep_unknown=(not args.no_unknown),
    )

    restore_splits(
        aligned_path=Path(args.str_aligned),
        outdir=outdir,
        prefix="str",
        keep_unknown=(not args.no_unknown),
    )

if __name__ == "__main__":
    main()
