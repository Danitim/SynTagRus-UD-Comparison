import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Iterator, Optional

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

@dataclass
class Token:
    sent_id: Optional[str]
    idx: int
    upos: str
    feats_norm: str

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
            "Убедись, что порядок предложений совпадает, и предсказание делалось на тех же входах."
        )

def accuracy(n_correct: int, n_total: int) -> float:
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
        "UPOS":   accuracy(upos_ok, n),
        "Feats":  accuracy(feats_ok, n),
        "AllTags":accuracy(all_ok, n),
        "Total":  n,
    }

def main():
    ap = argparse.ArgumentParser(description="Evaluate UPOS / FEATS / AllTags on CoNLL-U (gold vs pred).")
    ap.add_argument("gold", help="Путь к gold CoNLL-U")
    ap.add_argument("pred", help="Путь к предсказанному CoNLL-U")
    ap.add_argument("--show-mismatches", type=int, default=0,
                    help="Показать первые K несовпадений (для отладки)")
    args = ap.parse_args()

    scores = eval_morph(args.gold, args.pred)

    print("Metric   |   Accuracy")
    print("---------+-----------")
    print(f"UPOS     | {scores['UPOS']:9.2f}")
    print(f"Feats    | {scores['Feats']:9.2f}")
    print(f"AllTags  | {scores['AllTags']:9.2f}")
    print(f"(Total tokens compared: {scores['Total']})")

    if args.show_mismatches:
        gold = list(iter_words(args.gold))
        pred = list(iter_words(args.pred))
        shown = 0
        for g, p in zip(gold, pred):
            if (g.upos != p.upos) or (g.feats_norm != p.feats_norm):
                print("\n--- mismatch ---")
                print(f"sent_id: {g.sent_id}")
                print(f"UPOS: gold={g.upos} | pred={p.upos}")
                print(f"FEATS:\n  gold: {g.feats_norm or '_'}\n  pred: {p.feats_norm or '_'}")
                shown += 1
                if shown >= args.show_mismatches:
                    break

if __name__ == "__main__":
    main()
