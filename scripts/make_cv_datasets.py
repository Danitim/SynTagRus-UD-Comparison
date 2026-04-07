"""
Create 8-fold cross-validation dataset splits for mmBERT training.

The training set is split into 8 equal continuous chunks. Each fold uses
one chunk as the held-out "test", the remaining 7 chunks as "train", and
keeps the original dev.conllu unchanged.

Output structure:
  datasets_mmbert/<corpus>-cv<k>/train.conllu
  datasets_mmbert/<corpus>-cv<k>/dev.conllu
  datasets_mmbert/<corpus>-cv<k>/test.conllu
"""

import argparse
import shutil
from pathlib import Path


def read_sentences(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [b for b in text.split("\n\n") if b.strip()]


def write_sentences(path: Path, sentences: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sentences) + "\n\n", encoding="utf-8")


def make_cv_splits(
    datasets_root: Path,
    out_root: Path,
    corpora: list[str],
    n_folds: int,
) -> None:
    for corpus in corpora:
        src = datasets_root / corpus
        train_sents = read_sentences(src / "train.conllu")

        total = len(train_sents)
        base = total // n_folds
        remainder = total % n_folds

        print(f"{corpus}: {total} train sents, {n_folds} folds")

        # First `remainder` folds get one extra sentence to distribute evenly
        folds: list[list[str]] = []
        start = 0
        for i in range(n_folds):
            size = base + (1 if i < remainder else 0)
            folds.append(train_sents[start : start + size])
            start += size

        for k, held_out in enumerate(folds, start=1):
            fold_dir = out_root / f"{corpus}-cv{k}"

            train_for_fold = []
            for j, fold in enumerate(folds, start=1):
                if j != k:
                    train_for_fold.extend(fold)

            write_sentences(fold_dir / "train.conllu", train_for_fold)
            write_sentences(fold_dir / "test.conllu", held_out)
            shutil.copy(src / "dev.conllu", fold_dir / "dev.conllu")

            print(f"  fold {k}: train={len(train_for_fold)}, test={len(held_out)}")

        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-root",
        default="datasets",
        help="Source datasets directory (default: datasets)",
    )
    parser.add_argument(
        "--out-root",
        default="datasets_mmbert",
        help="Output datasets directory (default: datasets_mmbert)",
    )
    parser.add_argument(
        "--corpora",
        nargs="+",
        default=["str-new", "ud-new"],
        help="Corpora to process (default: str-new ud-new)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=8,
        help="Number of folds (default: 8)",
    )
    args = parser.parse_args()

    make_cv_splits(
        datasets_root=Path(args.datasets_root),
        out_root=Path(args.out_root),
        corpora=args.corpora,
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()
