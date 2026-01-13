import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

@dataclass
class Token:
    sent_id: Optional[str]
    idx: int
    upos: str
    feats_norm: str

def is_real_token(tok_form: str) -> bool:
    return (tok_form is not None) and (tok_form != '_')

def parse_feats(s: str) -> str:
    if not s or s == "_" or s.strip() == "":
        return ""
    parts = []
    for item in s.split("|"):
        item = item.strip()
        if not item or item == "_":
            continue
        if "=" not in item:
            key, val = item, ""
        else:
            key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key:
            parts.append((key, val))
    parts.sort(key=lambda kv: (kv[0], kv[1]))
    return "|".join(f"{k}={v}" if v != "" else k for k, v in parts)

def iter_words(conllu_path: str) -> Iterator[Token]:
    sent_id: Optional[str] = None
    with open(conllu_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                sent_id = None
                continue
            if line.startswith("#"):
                if line.lower().startswith("# sent_id") or line.startswith("# sent_id"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        sent_id = parts[1].strip()
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 10:
                raise ValueError(f"Некорректная строка (не 10 колонок): {line.strip()}")
            tok_form = cols[FORM]
            if not is_real_token(tok_form):
                continue
            tok_id = cols[ID]
            try:
                _ = int(tok_id)
            except ValueError:
                raise ValueError(f"Некорректный ID токена: {tok_id}")
            yield Token(
                sent_id=sent_id,
                idx=int(tok_id),
                upos=cols[UPOS],
                feats_norm=parse_feats(cols[FEATS]),
            )

def check_isomorphic(gold_tokens: List[Token], pred_tokens: List[Token]) -> None:
    if len(gold_tokens) != len(pred_tokens):
        raise RuntimeError(
            f"Несовпадение числа сравниваемых слов: gold={len(gold_tokens)}, pred={len(pred_tokens)}.\n"
        )

def _accuracy(n_correct: int, n_total: int) -> float:
    return 100.0 * n_correct / n_total if n_total else 0.0

def eval_morph(gold_path: str, pred_path: str) -> Dict[str, float]:
    gold = list(iter_words(gold_path))
    pred = list(iter_words(pred_path))
    check_isomorphic(gold, pred)

    n = len(gold)
    upos_ok = feats_ok = all_ok = 0

    for g, p in zip(gold, pred):
        u_ok = (g.upos == p.upos)
        f_ok = (g.feats_norm == p.feats_norm)
        upos_ok += int(u_ok)
        feats_ok += int(f_ok)
        all_ok  += int(u_ok and f_ok)

    return {
        "UPOS":    _accuracy(upos_ok, n),
        "Feats":   _accuracy(feats_ok, n),
        "AllTags": _accuracy(all_ok, n),
        "Total":   float(n),
    }

DISPLAY_NAMES: Dict[str, str] = {
    "ud":       "UD",
    "ud-old":   "UD-Old",
    "ud-new":   "UD-New",
    "str":      "SynTagRus",
    "str-old":  "SynTagRus-Old",
    "str-new":  "SynTagRus-New",
}

ORDER: List[str] = ["ud", "ud-new", "ud-old", "str", "str-new", "str-old"]

def block_header(title: str) -> str:
    return f"{title}:\nMetric   |   Accuracy\n---------+-----------\n"

def block_body(scores: Dict[str, float]) -> str:
    return (
        f"UPOS     | {scores['UPOS']:9.2f}\n"
        f"Feats    | {scores['Feats']:9.2f}\n"
        f"AllTags  | {scores['AllTags']:9.2f}\n"
        f"(Total tokens compared: {int(scores['Total'])})\n"
    )

def run_and_collect(
    datasets_root: Path,
    outputs_root: Path,
) -> List[str]:
    lines: List[str] = []
    for corpus in ORDER:
        title = DISPLAY_NAMES.get(corpus, corpus)

        gold_path = datasets_root / corpus / "test.conllu"
        pred_path = outputs_root / f"{corpus}-morph" / "test.pred.conllu"

        if not gold_path.is_file():
            print(f"[SKIP] No gold-file: {gold_path}")
            continue
        if not pred_path.is_file():
            print(f"[SKIP] No predictions: {pred_path}")
            continue

        try:
            scores = eval_morph(str(gold_path), str(pred_path))
        except Exception as e:
            print(f"[ERR] Error while evaluating {corpus}: {e}")
            continue

        lines.append(block_header(title))
        lines.append(block_body(scores))
        lines.append("")

    return lines

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate morphology for six corpora and write results.txt"
    )
    ap.add_argument(
        "--datasets-root", type=Path, default=Path("datasets"),
        help="Путь к корню gold-датасетов (по умолчанию: datasets)"
    )
    ap.add_argument(
        "--outputs-root", type=Path, default=Path("out"),
        help="Путь к корню предсказаний (по умолчанию: out)"
    )
    ap.add_argument(
        "--results-path", type=Path, default=Path("results.txt"),
        help="Куда записать сводный отчёт (по умолчанию: results.txt)"
    )
    args = ap.parse_args()

    lines = run_and_collect(args.datasets_root, args.outputs_root)

    if not lines:
        print("[WARN] No blocks collected — check paths to data and predictions.")
        return 1

    text = "\n".join(lines).rstrip() + "\n"
    args.results_path.write_text(text, encoding="utf-8")
    print(f"[OK] Results written to {args.results_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
