import argparse
from pathlib import Path


def read_sentences(path: Path) -> list[tuple[str | None, str]]:
    """Return list of (sent_id, raw_block) for each sentence."""
    text = path.read_text(encoding="utf-8")
    blocks = [b for b in text.split("\n\n") if b.strip()]
    result = []
    for block in blocks:
        sent_id = None
        for line in block.splitlines():
            if line.startswith("# sent_id"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    sent_id = parts[1].strip()
                break
        result.append((sent_id, block))
    return result


def merge_and_check(files: list[Path], out_path: Path) -> int:
    all_sentences: list[tuple[str | None, str]] = []
    seen_ids: set[str] = set()
    duplicates: list[str] = []

    for f in files:
        if not f.exists():
            print(f"  [skip] not found: {f}")
            continue
        sents = read_sentences(f)
        print(f"  [read] {f}: {len(sents)} sentences")
        for sent_id, block in sents:
            if sent_id is not None:
                if sent_id in seen_ids:
                    duplicates.append(sent_id)
                else:
                    seen_ids.add(sent_id)
            all_sentences.append((sent_id, block))

    if duplicates:
        print(f"  [ERROR] {len(duplicates)} duplicate sent_ids found:")
        for dup in duplicates[:20]:
            print(f"    - {dup}")
        if len(duplicates) > 20:
            print(f"    ... and {len(duplicates) - 20} more")
        raise ValueError(f"Duplicate sent_ids in {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n\n".join(block for _, block in all_sentences) + "\n\n",
        encoding="utf-8",
    )
    print(f"  [write] {out_path}: {len(all_sentences)} sentences total")
    return len(all_sentences)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", default="datasets",
                        help="Root with original datasets (default: datasets)")
    parser.add_argument("--cv-root", default="datasets_mmbert",
                        help="Root with CV datasets (default: datasets_mmbert)")
    parser.add_argument("--out-root", default="out/UDPipe2/mmBERT",
                        help="Root with predictions (default: out/UDPipe2/mmBERT)")
    parser.add_argument("--corpora", nargs="+", default=["str", "ud"],
                        help="Base corpus names (default: str ud)")
    parser.add_argument("--n-folds", type=int, default=9,
                        help="Number of CV folds (default: 9)")
    parser.add_argument("--num-runs", type=int, default=5,
                        help="Number of model runs (default: 5)")
    args = parser.parse_args()

    datasets_root = Path(args.datasets_root)
    cv_root = Path(args.cv_root)
    out_root = Path(args.out_root)

    for base in args.corpora:
        corpus = f"{base}-new"
        cv_corpora = [f"{corpus}-cv{k}" for k in range(1, args.n_folds + 1)]

        # --- Gold test files ---
        gold_files = [datasets_root / corpus / "test.conllu"] + \
                     [cv_root / cv / "test.conllu" for cv in cv_corpora]

        print(f"\n=== {corpus}: merging gold test files ===")
        gold_out = out_root / "run1" / f"{corpus}-merged" / "test.conllu"
        # Gold is the same for all runs — write once, reuse path for display
        merge_and_check(gold_files, gold_out)

        # --- Prediction files per run ---
        for run_idx in range(1, args.num_runs + 1):
            pred_files = [out_root / f"run{run_idx}" / corpus / "test.pred.conllu"] + \
                         [out_root / f"run{run_idx}" / cv / "test.pred.conllu"
                          for cv in cv_corpora]

            pred_out = out_root / f"run{run_idx}" / f"{corpus}-merged" / "test.pred.conllu"
            print(f"\n  run{run_idx}: merging predictions")
            merge_and_check(pred_files, pred_out)


if __name__ == "__main__":
    main()
