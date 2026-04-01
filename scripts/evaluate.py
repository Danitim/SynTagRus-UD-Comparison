import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)


def is_real_token(tok_form: str) -> bool:
    return bool(tok_form) and tok_form != "_"


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


@dataclass
class Token:
    sent_id: Optional[str]
    idx: int
    upos: str
    feats_norm: str


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
                raise ValueError(f"Invalid CONLLU line (expected 10 columns): {line.strip()}")
            tok_form = cols[FORM]
            if not is_real_token(tok_form):
                continue
            tok_id = cols[ID]
            try:
                _ = int(tok_id)
            except ValueError:
                raise ValueError(f"Invalid token ID: {tok_id}")
            yield Token(
                sent_id=sent_id,
                idx=int(tok_id),
                upos=cols[UPOS],
                feats_norm=parse_feats(cols[FEATS]),
            )


def check_isomorphic(gold_tokens: List[Token], pred_tokens: List[Token]) -> None:
    if len(gold_tokens) != len(pred_tokens):
        raise RuntimeError(
            f"Non-isomorphic data: gold={len(gold_tokens)}, pred={len(pred_tokens)}."
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
        all_ok += int(u_ok and f_ok)

    return {
        "UPOS": _accuracy(upos_ok, n),
        "Feats": _accuracy(feats_ok, n),
        "AllTags": _accuracy(all_ok, n),
        "Total": float(n),
    }


def _iter_sentences(path: str) -> Iterator[List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        sent: List[str] = []
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if sent:
                    yield sent
                    sent = []
                continue
            sent.append(line)
        if sent:
            yield sent


def _parse_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def _iter_syntax_tokens(sentence_lines: List[str]) -> Iterator[tuple]:
    for line in sentence_lines:
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 10:
            raise ValueError(f"Invalid CONLLU line (expected 10 columns): {line}")

        tok_id = cols[ID]
        if not tok_id.isdigit():
            continue

        head = _parse_int(cols[HEAD])
        if head is None:
            continue

        deprel = cols[DEPREL].split(":", 1)[0]
        yield (int(tok_id), head, deprel, cols[UPOS])


def _eval_ucm_lcm(gold_path: str, pred_path: str) -> Dict[str, float]:
    gold_sents = list(_iter_sentences(gold_path))
    pred_sents = list(_iter_sentences(pred_path))
    if len(gold_sents) != len(pred_sents):
        raise RuntimeError(f"Non-isomorphic data: sentences gold={len(gold_sents)} pred={len(pred_sents)}")

    n_sent = len(gold_sents)
    ucm_ok = 0
    lcm_ok = 0

    for gold_lines, pred_lines in zip(gold_sents, pred_sents):
        gold_tokens = list(_iter_syntax_tokens(gold_lines))
        pred_tokens = list(_iter_syntax_tokens(pred_lines))

        gold_map = {tid: (head, rel, upos) for tid, head, rel, upos in gold_tokens}
        pred_map = {tid: (head, rel) for tid, head, rel, _ in pred_tokens}

        gold_nonpunct_ids = [tid for tid, (_, _, upos) in gold_map.items() if upos != "PUNCT"]

        unlabeled_all = True
        labeled_all = True
        for tid in gold_nonpunct_ids:
            g_head, g_rel, _ = gold_map[tid]
            pred_entry = pred_map.get(tid)
            if pred_entry is None:
                unlabeled_all = False
                labeled_all = False
                break
            p_head, p_rel = pred_entry
            if g_head != p_head:
                unlabeled_all = False
                labeled_all = False
                break
            if g_rel != p_rel:
                labeled_all = False

        ucm_ok += int(unlabeled_all)
        lcm_ok += int(labeled_all)

    return {
        "UCM": 100.0 * ucm_ok / n_sent if n_sent else 0.0,
        "LCM": 100.0 * lcm_ok / n_sent if n_sent else 0.0,
    }


def _eval_ellipsis_uas_las(gold_path: str, pred_path: str) -> Dict[str, float]:
    """UAS/LAS for tokens whose gold head is an ellipsis node (form == '_')."""
    gold_sents = list(_iter_sentences(gold_path))
    pred_sents = list(_iter_sentences(pred_path))
    if len(gold_sents) != len(pred_sents):
        raise RuntimeError(f"Non-isomorphic data: sentences gold={len(gold_sents)} pred={len(pred_sents)}")

    total = 0
    uas_ok = 0
    las_ok = 0

    for gold_lines, pred_lines in zip(gold_sents, pred_sents):
        # Collect ellipsis node IDs from gold (form == "_")
        ellipsis_ids: set = set()
        for line in gold_lines:
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) != 10:
                continue
            if not cols[ID].isdigit():
                continue
            if not is_real_token(cols[FORM]):
                ellipsis_ids.add(int(cols[ID]))

        if not ellipsis_ids:
            continue

        # Build pred map: tok_id -> (head, deprel)
        pred_map: Dict[int, tuple] = {}
        for line in pred_lines:
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) != 10:
                continue
            if not cols[ID].isdigit():
                continue
            head = _parse_int(cols[HEAD])
            deprel = cols[DEPREL].split(":", 1)[0]
            pred_map[int(cols[ID])] = (head, deprel)

        # Evaluate regular tokens whose gold head is an ellipsis node
        for line in gold_lines:
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) != 10:
                continue
            if not cols[ID].isdigit():
                continue
            if not is_real_token(cols[FORM]):
                continue
            g_head = _parse_int(cols[HEAD])
            if g_head not in ellipsis_ids:
                continue
            g_rel = cols[DEPREL].split(":", 1)[0]
            pred_entry = pred_map.get(int(cols[ID]))

            total += 1
            if pred_entry is None:
                continue
            p_head, p_rel = pred_entry
            head_ok = (p_head == g_head)
            uas_ok += int(head_ok)
            las_ok += int(head_ok and p_rel == g_rel)

    return {
        "UAS_ELL": 100.0 * uas_ok / total if total else 0.0,
        "LAS_ELL": 100.0 * las_ok / total if total else 0.0,
    }


def eval_syntax(gold_path: str, pred_path: str) -> Dict[str, float]:
    import os
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "vendor" / "udpipe2"))
    import udpipe2_eval

    gold = udpipe2_eval.load_conllu_file(gold_path, single_root=1)
    pred = udpipe2_eval.load_conllu_file(pred_path, single_root=1)
    evaluation = udpipe2_eval.evaluate(gold, pred)

    scores: Dict[str, float] = {
        "UAS": 100.0 * evaluation["UAS"].f1,
        "LAS": 100.0 * evaluation["LAS"].f1,
    }
    scores.update(_eval_ucm_lcm(gold_path, pred_path))
    scores.update(_eval_ellipsis_uas_las(gold_path, pred_path))
    return scores


DISPLAY_NAMES: Dict[str, str] = {
    "ud": "UD",
    "ud-old": "UD-Old",
    "ud-new": "UD-New",
    "str": "SynTagRus",
    "str-old": "SynTagRus-Old",
    "str-new": "SynTagRus-New",
}

ORDER: List[str] = ["ud", "ud-new", "ud-old", "str", "str-new", "str-old"]


def block_header(title: str, task: str) -> str:
    if task == "morph":
        return f"{title}:\nMetric   |   Accuracy\n---------+-----------\n"
    return f"{title}:\nMetric |     Score\n-------+----------\n"


def block_body(scores: Dict[str, float], task: str) -> str:
    if task == "morph":
        return (
            f"UPOS     | {scores['UPOS']:9.2f}\n"
            f"Feats    | {scores['Feats']:9.2f}\n"
            f"AllTags  | {scores['AllTags']:9.2f}\n"
        )
    return "\n".join(f"{m:6} | {scores[m]:9.2f}" for m in ["UAS", "LAS", "UCM", "LCM", "UAS_ELL", "LAS_ELL"]) + "\n"


def run_and_collect(
    datasets_root: Path,
    outputs_root: Path,
    suffix: str,
    task: str,
) -> List[str]:
    lines: List[str] = []
    for corpus in ORDER:
        title = DISPLAY_NAMES.get(corpus, corpus)

        gold_path = datasets_root / corpus / "test.conllu"
        pred_path = outputs_root / f"{corpus}-{suffix}" / "test.pred.conllu"

        if not gold_path.is_file():
            print(f"[SKIP] No gold-file: {gold_path}")
            continue
        if not pred_path.is_file():
            print(f"[SKIP] No predictions: {pred_path}")
            continue

        try:
            if task == "morph":
                scores = eval_morph(str(gold_path), str(pred_path))
            else:
                scores = eval_syntax(str(gold_path), str(pred_path))
        except Exception as e:
            print(f"[ERR] Error while evaluating {corpus}: {e}")
            continue

        lines.append(block_header(title, task))
        lines.append(block_body(scores, task))
        lines.append("")

    return lines


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate morphology and/or syntax for six corpora based on suffix."
    )
    ap.add_argument("suffix", choices=["morph", "syntax", "morphsyntax"])
    ap.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    ap.add_argument("--outputs-root", type=Path, default=Path("out"))
    ap.add_argument("--results-path", type=Path, default=None)
    args = ap.parse_args()

    results_path = args.results_path or Path(f"results_{args.suffix}.txt")

    if args.suffix != "morphsyntax":
        task = "morph" if args.suffix == "morph" else "syntax"
        all_lines = run_and_collect(args.datasets_root, args.outputs_root, args.suffix, task)
    else:
        all_lines: List[str] = []
        for corpus in ORDER:
            title = DISPLAY_NAMES.get(corpus, corpus)
            gold_path = args.datasets_root / corpus / "test.conllu"
            pred_path = args.outputs_root / f"{corpus}-{args.suffix}" / "test.pred.conllu"

            if not gold_path.is_file():
                print(f"[SKIP] No gold-file: {gold_path}")
                continue
            if not pred_path.is_file():
                print(f"[SKIP] No predictions: {pred_path}")
                continue

            try:
                morph_scores = eval_morph(str(gold_path), str(pred_path))
                syntax_scores = eval_syntax(str(gold_path), str(pred_path))
            except Exception as e:
                print(f"[ERR] Error while evaluating {corpus}: {e}")
                continue

            all_lines.append(block_header(title, "morph"))
            all_lines.append(block_body(morph_scores, "morph"))
            all_lines.append(block_body(syntax_scores, "syntax"))
            all_lines.append("")

    if not all_lines:
        print("[WARN] No blocks collected — check paths to data and predictions.")
        return 1

    results_path.write_text("\n".join(all_lines).rstrip() + "\n", encoding="utf-8")
    print(f"[OK] Results written to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
