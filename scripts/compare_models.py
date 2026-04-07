import argparse
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy import stats

SCHEMES = {
    "UD annotation scheme": {
        "Full": "UD",
        "New":  "UD-New",
        "Old":  "UD-Old",
    },
    "SynTagRus annotation scheme": {
        "Full": "SynTagRus",
        "New":  "SynTagRus-New",
        "Old":  "SynTagRus-Old",
    },
}

METRICS = ["UPOS", "Feats", "AllTags", "UAS", "LAS", "UCM", "LCM", "UAS_ELL", "LAS_ELL"]

ALPHA = 0.05


def parse_raw_file(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    current_run: Optional[str] = None
    current_corpus: Optional[str] = None

    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^===\s*(\S+)\s*===$", line)
        if m:
            current_run = m.group(1)
            data[current_run] = {}
            current_corpus = None
            continue

        if current_run is None:
            continue

        m = re.match(r"^([A-Za-z][A-Za-z0-9\-]*):\s*$", line)
        if m:
            current_corpus = m.group(1)
            data[current_run].setdefault(current_corpus, {})
            continue

        if current_corpus is None:
            continue

        m = re.match(r"^(\w+)\s*\|\s*([\d.]+)\s*$", line)
        if m:
            data[current_run][current_corpus][m.group(1)] = float(m.group(2))

    return data


def paired_ttest(
    a_vals: List[float], b_vals: List[float]
) -> Tuple[float, float, float, float, float]:
    mean_a = statistics.mean(a_vals)
    std_a  = statistics.stdev(a_vals) if len(a_vals) > 1 else 0.0
    mean_b = statistics.mean(b_vals)
    std_b  = statistics.stdev(b_vals) if len(b_vals) > 1 else 0.0
    _, p   = stats.ttest_rel(a_vals, b_vals)
    return mean_a, std_a, mean_b, std_b, p


def fmt_cell_plain(mean: float, std: float) -> str:
    return f"{mean:6.2f} ±{std:.2f}"


def fmt_delta_plain(delta: float, sig: bool) -> str:
    marker = "*" if sig else " "
    return f"{delta:+.2f}{marker}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a", type=Path)
    ap.add_argument("file_b", type=Path)
    ap.add_argument("--model-a", default="A")
    ap.add_argument("--model-b", default="B")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()

    data_a = parse_raw_file(args.file_a)
    data_b = parse_raw_file(args.file_b)

    runs_a = sorted(data_a.keys())
    runs_b = sorted(data_b.keys())
    if runs_a != runs_b:
        print(f"[WARN] Run lists differ: {runs_a} vs {runs_b}")

    runs = [r for r in runs_a if r in data_b]
    print(f"[INFO] Paired runs: {runs}")

    results: Dict = {}

    for scheme, cols in SCHEMES.items():
        results[scheme] = {}
        for col_label, corpus in cols.items():
            results[scheme][col_label] = {}
            for metric in METRICS:
                vals_a = [data_a[r].get(corpus, {}).get(metric) for r in runs]
                vals_b = [data_b[r].get(corpus, {}).get(metric) for r in runs]
                vals_a = [v for v in vals_a if v is not None]
                vals_b = [v for v in vals_b if v is not None]
                if not vals_a or not vals_b or len(vals_a) != len(vals_b):
                    results[scheme][col_label][metric] = None
                    continue
                results[scheme][col_label][metric] = paired_ttest(vals_a, vals_b)

    cols_order = ["Full", "New", "Old"]
    a, b = args.model_a, args.model_b

    plain_lines: List[str] = []
    col_w = 16
    delta_w = 8

    header1 = f"{'Metric':<10}"
    header2 = f"{'':10}"
    for col in cols_order:
        header1 += f"  {col:<{col_w * 2 + delta_w + 4}}"
        header2 += f"  {a:<{col_w}}  {b:<{col_w}}  {'Δ (*)':>{delta_w}}"
    plain_lines += [header1, header2, "-" * len(header2)]

    for scheme, col_data in results.items():
        plain_lines.append(f"\n  {scheme}")
        for metric in METRICS:
            row = f"{metric:<10}"
            for col in cols_order:
                entry = col_data[col].get(metric)
                if entry is None:
                    row += f"  {'—':>{col_w}}  {'—':>{col_w}}  {'—':>{delta_w}}"
                else:
                    mean_a, std_a, mean_b, std_b, p = entry
                    sig = p < args.alpha
                    row += (
                        f"  {fmt_cell_plain(mean_a, std_a):>{col_w}}"
                        f"  {fmt_cell_plain(mean_b, std_b):>{col_w}}"
                        f"  {fmt_delta_plain(mean_b - mean_a, sig):>{delta_w}}"
                    )
            plain_lines.append(row)

    plain_lines.append(f"\n* p < {args.alpha} (paired t-test, two-tailed)")

    out_base = args.output or Path(f"out/comparison_{a}_vs_{b}")
    out_base.parent.mkdir(parents=True, exist_ok=True)

    txt_path = out_base.with_suffix(".txt")

    txt_path.write_text("\n".join(plain_lines) + "\n", encoding="utf-8")

    print(f"[OK] {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
