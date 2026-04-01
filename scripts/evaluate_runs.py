import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import DISPLAY_NAMES, ORDER, eval_morph, eval_syntax

MORPH_METRICS = ["UPOS", "Feats", "AllTags"]
SYNTAX_METRICS = ["UAS", "LAS", "UCM", "LCM", "UAS_ELL", "LAS_ELL"]


def find_runs(runs_root: Path) -> List[Path]:
    runs = sorted(
        [d for d in runs_root.iterdir() if d.is_dir() and d.name.startswith("run")],
        key=lambda p: p.name,
    )
    return runs


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def format_metric_row(name: str, values: List[float], col_width: int = 8) -> str:
    m, s = mean_std(values)
    n = len(values)
    return f"{name:<{col_width}} | {m:6.2f} ± {s:.2f}  (n={n})"


def collect_corpus_scores(
    runs: List[Path],
    corpus: str,
    gold_path: Path,
    task: str,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Returns (morph_scores_per_metric, syntax_scores_per_metric)."""
    morph: Dict[str, List[float]] = {m: [] for m in MORPH_METRICS}
    syntax: Dict[str, List[float]] = {m: [] for m in SYNTAX_METRICS}

    for run_dir in runs:
        pred_path = run_dir / corpus / "test.pred.conllu"
        if not pred_path.is_file():
            print(f"[SKIP] Not found: {pred_path}")
            continue
        try:
            if task in ("morph", "morphsyntax"):
                s = eval_morph(str(gold_path), str(pred_path))
                for m in MORPH_METRICS:
                    morph[m].append(s[m])
            if task in ("syntax", "morphsyntax"):
                s = eval_syntax(str(gold_path), str(pred_path))
                for m in SYNTAX_METRICS:
                    syntax[m].append(s[m])
        except Exception as e:
            print(f"[ERR] {run_dir.name}/{corpus}: {e}")

    return morph, syntax


def format_block(
    title: str,
    task: str,
    morph: Dict[str, List[float]],
    syntax: Dict[str, List[float]],
) -> List[str]:
    lines: List[str] = []
    lines.append(f"{title}:")

    if task in ("morph", "morphsyntax"):
        if any(morph[m] for m in MORPH_METRICS):
            lines.append(f"{'Metric':<8} | Mean ± Std")
            lines.append("-" * 9 + "+" + "-" * 22)
            for m in MORPH_METRICS:
                if morph[m]:
                    lines.append(format_metric_row(m, morph[m]))

    if task in ("syntax", "morphsyntax"):
        if any(syntax[m] for m in SYNTAX_METRICS):
            lines.append("-" * 9 + "+" + "-" * 22)
            for m in SYNTAX_METRICS:
                if syntax[m]:
                    lines.append(format_metric_row(m, syntax[m]))

    return lines


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate multiple UDPipe2 runs and report mean ± std per corpus."
    )
    ap.add_argument("runs_root", type=Path, help="Root dir containing run1/, run2/, ...")
    ap.add_argument("task", choices=["morph", "syntax", "morphsyntax"])
    ap.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    ap.add_argument("--results-path", type=Path, default=None)
    args = ap.parse_args()

    if not args.runs_root.is_dir():
        print(f"[ERR] Not a directory: {args.runs_root}")
        return 1

    runs = find_runs(args.runs_root)
    if not runs:
        print(f"[ERR] No run* directories found in {args.runs_root}")
        return 1

    print(f"[INFO] Found {len(runs)} run(s): {[r.name for r in runs]}")

    results_path: Path = args.results_path or (args.runs_root / f"results_{args.task}.txt")

    all_lines: List[str] = []

    for corpus in ORDER:
        gold_path = args.datasets_root / corpus / "test.conllu"
        if not gold_path.is_file():
            print(f"[SKIP] No gold file: {gold_path}")
            continue

        morph, syntax = collect_corpus_scores(runs, corpus, gold_path, args.task)

        has_any = any(morph[m] for m in MORPH_METRICS) or any(syntax[m] for m in SYNTAX_METRICS)
        if not has_any:
            print(f"[SKIP] No predictions collected for corpus: {corpus}")
            continue

        title = DISPLAY_NAMES.get(corpus, corpus)
        block = format_block(title, args.task, morph, syntax)
        all_lines.extend(block)
        all_lines.append("")

    if not all_lines:
        print("[WARN] No results collected — check paths.")
        return 1

    results_path.write_text("\n".join(all_lines).rstrip() + "\n", encoding="utf-8")
    print(f"[OK] Results written to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
