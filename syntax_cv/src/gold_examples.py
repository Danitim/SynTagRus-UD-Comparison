"""
Utilities for using manually checked LaTeX examples as an internal gold set.

The files in examples/certified/correct/ are visual artefacts, not a public
training resource.  We use only their connector lines as regression checks:
for each rendered UD token, the current correspondence must point to the same
STR token as the manually accepted picture.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

import pandas as pd

from src.edge_correspondence import EdgeCorrespondence, SentID


_DRAW_RE = re.compile(r"str-w-(\d+).*?ud-w-(\d+)")


def extract_tex_token_links(
    tex_path: str | Path,
    str_df: pd.DataFrame,
    ud_df: pd.DataFrame,
    sent_id: Optional[SentID] = None,
) -> dict[int, int]:
    """
    Parse a rendered .tex example into {ud_token_id: str_token_id}.

    The renderer names nodes by token column (`str-w-1`, `ud-w-1`), not by
    token id.  We therefore map columns back to ids using the sentence tables.
    """
    tex_path = Path(tex_path)
    sent_id = tex_path.stem if sent_id is None else sent_id

    str_col_to_id = _column_to_token_id(str_df, sent_id)
    ud_col_to_id = _column_to_token_id(ud_df, sent_id)

    links: dict[int, int] = {}
    for str_col_s, ud_col_s in _DRAW_RE.findall(tex_path.read_text(encoding="utf-8")):
        str_id = str_col_to_id.get(int(str_col_s))
        ud_id = ud_col_to_id.get(int(ud_col_s))
        if str_id is None or ud_id is None:
            continue
        links[int(ud_id)] = int(str_id)
    return links


def current_token_links(
    str_df: pd.DataFrame,
    ud_df: pd.DataFrame,
    corr: EdgeCorrespondence,
    sent_id: SentID,
    *,
    include_root: bool = True,
) -> dict[int, int]:
    """
    Return the current rendered token links as {ud_token_id: str_token_id}.

    This mirrors `tikz_dep.render_pair`: punctuation links are ignored and an
    extra root connector is included when requested.
    """
    str_view = _sentence_view(str_df, sent_id)
    ud_view = _sentence_view(ud_df, sent_id)
    str_punct = _punct_ids(str_view)
    ud_punct = _punct_ids(ud_view)

    out: dict[int, int] = {}
    sc = corr.per_sentence.get(sent_id)
    if sc is not None:
        ud_sk = corr.ud_skel[sent_id]
        str_sk = corr.str_skel[sent_id]
        for e_ud, match in sc.matches.items():
            if match.e_str is None:
                continue
            dep_ud = ud_sk.dep_of(e_ud)
            dep_str = str_sk.dep_of(match.e_str)
            if dep_ud is None or dep_str is None:
                continue
            if dep_ud in ud_punct or dep_str in str_punct:
                continue
            out[int(dep_ud)] = int(dep_str)

    if include_root:
        str_roots = _root_ids(str_view)
        ud_roots = _root_ids(ud_view)
        if str_roots and ud_roots:
            out[int(sorted(ud_roots)[0])] = int(sorted(str_roots)[0])

    return out


def validate_gold_tex_examples(
    str_df: pd.DataFrame,
    ud_df: pd.DataFrame,
    corr: EdgeCorrespondence,
    correct_dir: str | Path = "examples/certified/correct",
) -> pd.DataFrame:
    """
    Compare current correspondence with all manually checked .tex files.

    Returns one row per expected or extra current connector.  `ok` is True
    only when the current STR token equals the manually accepted STR token.
    """
    correct_dir = Path(correct_dir)
    rows: list[dict[str, object]] = []

    for tex_path in sorted(correct_dir.glob("*.tex")):
        sent_id = tex_path.stem
        expected = extract_tex_token_links(tex_path, str_df, ud_df, sent_id)
        actual = current_token_links(str_df, ud_df, corr, sent_id)
        str_forms = _form_lookup(str_df, sent_id)
        ud_forms = _form_lookup(ud_df, sent_id)

        for ud_id in sorted(set(expected) | set(actual)):
            expected_str = expected.get(ud_id)
            actual_str = actual.get(ud_id)
            rows.append(
                {
                    "sent_id": sent_id,
                    "ud_id": ud_id,
                    "ud_form": ud_forms.get(ud_id),
                    "expected_str_id": expected_str,
                    "expected_str_form": str_forms.get(expected_str),
                    "actual_str_id": actual_str,
                    "actual_str_form": str_forms.get(actual_str),
                    "ok": expected_str == actual_str,
                    "source_file": str(tex_path),
                }
            )

    return pd.DataFrame(rows)


def _sentence_view(df: pd.DataFrame, sent_id: SentID) -> pd.DataFrame:
    out = df
    if "sent_id" not in out.columns or "id" not in out.columns:
        out = out.reset_index()
    if "head" not in out.columns and "head_g" in out.columns:
        out = out.assign(head=out["head_g"])
    if "deprel" not in out.columns and "deprel_g" in out.columns:
        out = out.assign(deprel=out["deprel_g"])
    return out[out["sent_id"] == sent_id].sort_values("id").reset_index(drop=True)


def _column_to_token_id(df: pd.DataFrame, sent_id: SentID) -> dict[int, int]:
    toks = _sentence_view(df, sent_id)
    return {i + 1: int(row["id"]) for i, row in toks.iterrows()}


def _punct_ids(tokens: pd.DataFrame) -> set[int]:
    out: set[int] = set()
    for _, row in tokens.iterrows():
        deprel = str(row.get("deprel", "")).lower()
        upos = str(row.get("upos", "")).upper()
        if deprel == "punct" or upos == "PUNCT":
            out.add(int(row["id"]))
    return out


def _root_ids(tokens: pd.DataFrame) -> set[int]:
    return {
        int(row["id"])
        for _, row in tokens.iterrows()
        if int(row.get("head", row.get("head_g", -1))) == 0
    }


def _form_lookup(df: pd.DataFrame, sent_id: SentID) -> dict[int, object]:
    toks = _sentence_view(df, sent_id)
    if "form" not in toks.columns:
        return {}
    return {int(row["id"]): row["form"] for _, row in toks.iterrows()}
