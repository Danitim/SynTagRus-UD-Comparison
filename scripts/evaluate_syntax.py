import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "vendor", "udpipe2"))

import udpipe2_eval

SYNTAX_METRICS = ["UAS", "LAS", "UCM", "LCM", "UAS_ELL", "LAS_ELL"]


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


def _iter_syntax_tokens(sentence_lines: List[str]) -> Iterator[Tuple[int, str, int, str]]:
    # Yields (id, upos, head, deprel_base) for tokens included in attachment metrics.
    for line in sentence_lines:
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 10:
            raise ValueError(f"Invalid CONLLU line (expected 10 columns): {line}")

        tok_id = cols[udpipe2_eval.ID]
        if not tok_id.isdigit():
            continue  # ignore multi-word tokens and empty nodes

        upos = cols[udpipe2_eval.UPOS]
        if upos == "PUNCT":
            continue

        head = _parse_int(cols[udpipe2_eval.HEAD])
        if head is None:
            continue

        deprel = cols[udpipe2_eval.DEPREL].split(":", 1)[0]
        yield (int(tok_id), upos, head, deprel)


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
        if len(gold_tokens) != len(pred_tokens):
            raise RuntimeError(
                f"Non-isomorphic data: tokens gold={len(gold_tokens)} pred={len(pred_tokens)}"
            )

        # Require identical token ids in the considered subset.
        for (gi, _, _, _), (pi, _, _, _) in zip(gold_tokens, pred_tokens):
            if gi != pi:
                raise RuntimeError(f"Non-isomorphic data: token id mismatch gold={gi} pred={pi}")

        unlabeled_all = True
        labeled_all = True
        for (_, _, g_head, g_rel), (_, _, p_head, p_rel) in zip(gold_tokens, pred_tokens):
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


def _iter_ellipsis_tokens(sentence_lines: List[str]) -> Iterator[Tuple[int, Optional[int], str]]:
    # Yields (id, head, deprel_base) for ellipsis tokens.
    for line in sentence_lines:
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 10:
            raise ValueError(f"Invalid CONLLU line (expected 10 columns): {line}")

        tok_id = cols[udpipe2_eval.ID]
        if not tok_id.isdigit():
            continue  # ignore multi-word tokens and empty nodes

        misc = cols[udpipe2_eval.MISC]
        if not misc or misc == "_":
            continue
        misc_parts = [part.strip() for part in misc.split("|")]
        if "Ellipsis=Yes" not in misc_parts:
            continue

        head = _parse_int(cols[udpipe2_eval.HEAD])
        deprel = cols[udpipe2_eval.DEPREL].split(":", 1)[0]
        yield (int(tok_id), head, deprel)


def _eval_ellipsis_uas_las(gold_path: str, pred_path: str) -> Dict[str, float]:
    gold_sents = list(_iter_sentences(gold_path))
    pred_sents = list(_iter_sentences(pred_path))
    if len(gold_sents) != len(pred_sents):
        raise RuntimeError(f"Non-isomorphic data: sentences gold={len(gold_sents)} pred={len(pred_sents)}")

    total = 0
    uas_ok = 0
    las_ok = 0

    for gold_lines, pred_lines in zip(gold_sents, pred_sents):
        gold_tokens = list(_iter_ellipsis_tokens(gold_lines))
        pred_tokens = list(_iter_ellipsis_tokens(pred_lines))
        if len(gold_tokens) != len(pred_tokens):
            raise RuntimeError(
                f"Non-isomorphic data: ellipsis tokens gold={len(gold_tokens)} pred={len(pred_tokens)}"
            )

        for (gi, g_head, g_rel), (pi, p_head, p_rel) in zip(gold_tokens, pred_tokens):
            if gi != pi:
                raise RuntimeError(f"Non-isomorphic data: token id mismatch gold={gi} pred={pi}")

            total += 1
            head_ok = (g_head is not None) and (p_head is not None) and (g_head == p_head)
            uas_ok += int(head_ok)
            las_ok += int(head_ok and (g_rel == p_rel))

    return {
        "UAS_ELL": 100.0 * uas_ok / total if total else 0.0,
        "LAS_ELL": 100.0 * las_ok / total if total else 0.0,
    }


def eval_syntax(gold_path: str, pred_path: str) -> Dict[str, float]:
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


def block_header(title: str) -> str:
    return f"{title}:\nMetric |     Score\n-------+----------"


def block_body(scores: Dict[str, float]) -> str:
    return "\n".join(f"{m:6} | {scores[m]:9.2f}" for m in SYNTAX_METRICS) + "\n"


def run_and_collect(
    datasets_root: Path,
    outputs_root: Path,
) -> List[str]:
    lines: List[str] = []
    for corpus in ORDER:
        title = DISPLAY_NAMES.get(corpus, corpus)

        gold_path = datasets_root / corpus / "test.conllu"
        pred_path = outputs_root / f"{corpus}-syntax" / "test.pred.conllu"

        if not gold_path.is_file():
            print(f"[SKIP] No gold-file: {gold_path}")
            continue
        if not pred_path.is_file():
            print(f"[SKIP] No predictions: {pred_path}")
            continue

        try:
            scores = eval_syntax(str(gold_path), str(pred_path))
        except Exception as e:
            print(f"[ERR] Error while evaluating {corpus}: {e}")
            continue

        lines.append(block_header(title))
        lines.append(block_body(scores))
        lines.append("")

    return lines


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate UDPipe 2 syntax metrics for six corpora and write results_syntax.txt"
    )
    ap.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    ap.add_argument("--outputs-root", type=Path, default=Path("out"))
    ap.add_argument("--results-path", type=Path, default=Path("results_syntax.txt"))
    args = ap.parse_args()

    lines = run_and_collect(args.datasets_root, args.outputs_root)
    if not lines:
        print("[WARN] No blocks collected; check paths to data and predictions.")
        return 1

    args.results_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[OK] Results written to {args.results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
