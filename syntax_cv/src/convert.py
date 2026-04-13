import re
from pathlib import Path
from typing import List, Optional
import pandas as pd
from dataclasses import dataclass


_SENT_ID_RE = re.compile(r"^#\s*sent_id\s*=\s*(.*)$", flags=re.IGNORECASE)
_TEXT_RE = re.compile(r"^#\s*text\s*=\s*(.*)$", flags=re.IGNORECASE)
_ELLIPSIS_RE = re.compile(r"(?:^|\\|)Ellipsis=Yes(?:\\||$)")


@dataclass
class ConlluToken:
    ID: int
    FORM: str
    LEMMA: str
    UPOS: str
    XPOS: str
    FEATS: str
    HEAD: str
    DEPREL: str
    MISC: str


@dataclass
class ConlluSentence:
    tokens: List[ConlluToken]
    comments: List[str]
    text: Optional[str] = None


def read_conllu(path: str) -> List[ConlluSentence]:
    sents: List[ConlluSentence] = []
    cur_tokens: List[ConlluToken] = []
    cur_comments: List[str] = []
    cur_text: Optional[str] = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur_tokens:
                    sents.append(ConlluSentence(cur_tokens, cur_comments, cur_text))
                cur_tokens, cur_comments, cur_text = [], [], None
                continue

            if line.startswith("#"):
                cur_comments.append(line)
                m = _TEXT_RE.match(line)
                if m:
                    cur_text = m.group(1)
                continue

            cols = line.split("\t")
            if len(cols) < 10:
                continue

            tid = cols[0]
            try:
                tid_int = int(tid)
            except ValueError:
                continue

            tok = ConlluToken(
                ID=tid_int,
                FORM=cols[1],
                LEMMA=cols[2],
                UPOS=cols[3],
                XPOS=cols[4],
                FEATS=cols[5],
                HEAD=cols[6],
                DEPREL=cols[7],
                MISC=cols[9] if len(cols) > 9 else "",
            )
            cur_tokens.append(tok)

    if cur_tokens:
        sents.append(ConlluSentence(cur_tokens, cur_comments, cur_text))

    return sents


def reconstruct_sentence_text(tokens: List[ConlluToken]) -> str:
    return " ".join(t.FORM for t in tokens)


def _extract_sent_id(comments: List[str]) -> Optional[str]:
    for c in comments:
        m = _SENT_ID_RE.match(c)
        if m:
            return m.group(1).strip()
    return None


def _has_ellipsis_flag(misc: str) -> bool:
    if misc is None or misc == "" or misc == "_":
        return False
    return _ELLIPSIS_RE.search(str(misc)) is not None


def conllu_to_csv(
    conllu_path: str,
    csv_path: str,
    *,
    reconstruct_text_if_missing: bool = True,
) -> None:
    sents = read_conllu(conllu_path)
    rows_out = []
    for s in sents:
        sent_id = _extract_sent_id(s.comments)
        text = s.text
        if not text and reconstruct_text_if_missing:
            text = reconstruct_sentence_text(s.tokens)

        for tok in s.tokens:
            ellipsis = 1 if _has_ellipsis_flag(tok.MISC) else 0
            rows_out.append({
                "sent_id": sent_id,
                "text": text,
                "id": tok.ID,
                "form": tok.FORM,
                "upos": tok.UPOS if tok.UPOS != "_" else "",
                "feats": "" if tok.FEATS in (None, "", "_") else tok.FEATS,
                "head": tok.HEAD,
                "deprel": tok.DEPREL,
                "ellipsis": ellipsis,
            })

    df = pd.DataFrame(
        rows_out,
        columns=["sent_id", "text", "id", "form", "upos", "feats", "head", "deprel", "ellipsis"],
    )
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")


def convert_cv_to_csv(
    out_root: str,
    csv_dir: str,
    corpora: List[str],
    num_runs: int = 5,
    *,
    force: bool = False,
) -> dict:
    """
    Convert merged CV conllu files to CSV.

    Gold (same across runs) → {corpus}.test.csv
    Predictions per run    → {corpus}.test.pred.run{N}.csv

    Returns a dict:
      {corpus: {"gold": Path, "pred": {run_idx: Path}}}
    """
    out_root = Path(out_root)
    csv_dir = Path(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    index: dict = {}

    for corpus in corpora:
        merged = f"{corpus}-merged"

        # Gold — taken from run1 (identical across runs)
        gold_conllu = out_root / "run1" / merged / "test.conllu"
        gold_csv = csv_dir / f"{merged}.test.csv"

        if not gold_csv.exists() or force:
            if gold_conllu.exists():
                print(f"[convert] {gold_conllu} -> {gold_csv}")
                conllu_to_csv(str(gold_conllu), str(gold_csv))
            else:
                print(f"[skip] not found: {gold_conllu}")

        pred_csvs: dict = {}
        for run_idx in range(1, num_runs + 1):
            pred_conllu = out_root / f"run{run_idx}" / merged / "test.pred.conllu"
            pred_csv = csv_dir / f"{merged}.test.pred.run{run_idx}.csv"

            if not pred_csv.exists() or force:
                if pred_conllu.exists():
                    print(f"[convert] {pred_conllu} -> {pred_csv}")
                    conllu_to_csv(str(pred_conllu), str(pred_csv))
                else:
                    print(f"[skip] not found: {pred_conllu}")

            if pred_csv.exists():
                pred_csvs[run_idx] = pred_csv

        index[corpus] = {
            "gold": gold_csv if gold_csv.exists() else None,
            "pred": pred_csvs,
        }

    return index
