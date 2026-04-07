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
    return sorted(
        [d for d in runs_root.iterdir() if d.is_dir() and d.name.startswith("run")],
        key=lambda p: p.name,
    )


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def format_metric_row(name: str, value: float, col_width: int = 8) -> str:
    return f"{name:<{col_width}} | {value:6.2f}"


def format_metric_row_agg(name: str, values: List[float], col_width: int = 8) -> str:
    m, s = mean_std(values)
    return f"{name:<{col_width}} | {m:6.2f} ± {s:.2f}  (n={len(values)})"


def format_raw_block(
    title: str,
    task: str,
    morph: Dict[str, float],
    syntax: Dict[str, float],
) -> List[str]:
    lines = [f"{title}:"]
    if task in ("morph", "morphsyntax"):
        has = any(v for v in morph.values())
        if has:
            lines.append(f"{'Metric':<8} | Value")
            lines.append("-" * 9 + "+" + "-" * 10)
            for m in MORPH_METRICS:
                if morph.get(m) is not None:
                    lines.append(format_metric_row(m, morph[m]))
    if task in ("syntax", "morphsyntax"):
        has = any(v for v in syntax.values())
        if has:
            lines.append(f"{'Metric':<8} | Value")
            lines.append("-" * 9 + "+" + "-" * 10)
            for m in SYNTAX_METRICS:
                if syntax.get(m) is not None:
                    lines.append(format_metric_row(m, syntax[m]))
    return lines


def format_agg_block(
    title: str,
    task: str,
    morph: Dict[str, List[float]],
    syntax: Dict[str, List[float]],
) -> List[str]:
    lines = [f"{title}:"]
    if task in ("morph", "morphsyntax"):
        if any(morph[m] for m in MORPH_METRICS):
            lines.append(f"{'Metric':<8} | Mean ± Std")
            lines.append("-" * 9 + "+" + "-" * 22)
            for m in MORPH_METRICS:
                if morph[m]:
                    lines.append(format_metric_row_agg(m, morph[m]))
    if task in ("syntax", "morphsyntax"):
        if any(syntax[m] for m in SYNTAX_METRICS):
            lines.append("-" * 9 + "+" + "-" * 22)
            for m in SYNTAX_METRICS:
                if syntax[m]:
                    lines.append(format_metric_row_agg(m, syntax[m]))
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_root", type=Path)
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

    # raw_data[run_dir][corpus] = {"morph": {metric: float}, "syntax": {metric: float}}
    raw_data: Dict[Path, Dict[str, Dict]] = {r: {} for r in runs}

    # agg_data[corpus] = {"morph": {metric: [float]}, "syntax": {metric: [float]}}
    agg_data: Dict[str, Dict] = {}

    for corpus in ORDER:
        gold_path = args.datasets_root / corpus / "test.conllu"
        if not gold_path.is_file():
            print(f"[SKIP] No gold file: {gold_path}")
            continue

        agg_data[corpus] = {
            "morph": {m: [] for m in MORPH_METRICS},
            "syntax": {m: [] for m in SYNTAX_METRICS},
        }

        for run_dir in runs:
            pred_path = run_dir / corpus / "test.pred.conllu"
            if not pred_path.is_file():
                print(f"[SKIP] Not found: {pred_path}")
                raw_data[run_dir][corpus] = {"morph": {}, "syntax": {}}
                continue

            morph_scores: Dict[str, float] = {}
            syntax_scores: Dict[str, float] = {}

            try:
                if args.task in ("morph", "morphsyntax"):
                    s = eval_morph(str(gold_path), str(pred_path))
                    morph_scores = {m: s[m] for m in MORPH_METRICS}
                    for m in MORPH_METRICS:
                        agg_data[corpus]["morph"][m].append(s[m])
                if args.task in ("syntax", "morphsyntax"):
                    s = eval_syntax(str(gold_path), str(pred_path))
                    syntax_scores = {m: s[m] for m in SYNTAX_METRICS}
                    for m in SYNTAX_METRICS:
                        agg_data[corpus]["syntax"][m].append(s[m])
            except Exception as e:
                print(f"[ERR] {run_dir.name}/{corpus}: {e}")

            raw_data[run_dir][corpus] = {"morph": morph_scores, "syntax": syntax_scores}

    raw_lines: List[str] = []
    for run_dir in runs:
        run_lines: List[str] = []
        for corpus in ORDER:
            if corpus not in raw_data[run_dir]:
                continue
            scores = raw_data[run_dir][corpus]
            if not scores["morph"] and not scores["syntax"]:
                continue
            title = DISPLAY_NAMES.get(corpus, corpus)
            run_lines.extend(format_raw_block(title, args.task, scores["morph"], scores["syntax"]))
            run_lines.append("")

        if run_lines:
            raw_lines.append(f"=== {run_dir.name} ===")
            raw_lines.extend(run_lines)

    agg_lines: List[str] = []
    for corpus in ORDER:
        if corpus not in agg_data:
            continue
        title = DISPLAY_NAMES.get(corpus, corpus)
        block = format_agg_block(
            title, args.task, agg_data[corpus]["morph"], agg_data[corpus]["syntax"]
        )
        if len(block) > 1:
            agg_lines.extend(block)
            agg_lines.append("")

    if not agg_lines:
        print("[WARN] No results collected.")
        return 1

    agg_path = args.results_path or (args.runs_root / f"results_{args.task}.txt")
    agg_path.write_text("\n".join(agg_lines).rstrip() + "\n", encoding="utf-8")

    if raw_lines:
        raw_path = agg_path.with_stem(agg_path.stem + "_raw")
        raw_path.write_text("\n".join(raw_lines).rstrip() + "\n", encoding="utf-8")
        print(f"[OK] {raw_path}")
    print(f"[OK] {agg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
