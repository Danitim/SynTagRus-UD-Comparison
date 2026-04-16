"""
token_comparison.py — derive per-token label comparison from an
EdgeCorrespondence WITHOUT modifying either input DataFrame.

Replaces the old `apply_swap_plan` / `apply_token_mapping` pipeline, which
used to reshape the STR DataFrame so that token indices aligned with UD.
That reshape was convenient for naive `df["deprel_g_str"] == df["deprel_g_ud"]`
joins but had two fatal flaws:

  1. It destroyed the original STR syntax (e.g. turned a dependent into a
     head), so any downstream analysis of "the STR tree as an object"
     became unsound.
  2. When the swap graph had intersecting pairs it occasionally introduced
     edges that do not exist in either gold tree (the "надо/бы" analytical
     link discussed in the 16-Apr correspondence).

The functions in this module produce a read-only comparison table. Every
row answers exactly one question: for this UD token v, which STR token
should carry the comparable label, and what is the status of the match?
No column of `str_df` or `ud_df` is ever written.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from syntax_cv.src.edge_correspondence import (
    EdgeCorrespondence,
    SentID,
    TokenID,
)


# ---------------------------------------------------------------------------
# Main table builder
# ---------------------------------------------------------------------------

def build_comparison_table(
    str_df: pd.DataFrame,
    ud_df: pd.DataFrame,
    corr: EdgeCorrespondence,
    *,
    exclude_punct: bool = True,
    str_label_col: str = "deprel",
    ud_label_col: str = "deprel",
    extra_str_cols: tuple[str, ...] = (),
    extra_ud_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """
    Produce a long DataFrame of per-token comparisons.

    Columns (always present):
      sent_id, ud_id, ud_form, ud_head, ud_deprel,
      str_id, str_form, str_head, str_deprel,
      status, comparable

    `status` takes values from EdgeCorrespondence; `comparable` is a
    convenience boolean (True iff status in {exact_same_dir,
    exact_mirrored, restructured}).

    Parameters
    ----------
    str_df, ud_df    : unchanged input DataFrames (must include columns
                       sent_id, id, head, deprel, form; upos if
                       exclude_punct=True).
    corr             : EdgeCorrespondence built from the same two
                       DataFrames.
    exclude_punct    : drop rows where UD-side token has deprel=="punct"
                       or upos=="PUNCT". Matches standard UD evaluation.
    str_label_col    : column in `str_df` to use as the STR label. Default
                       "deprel"; set to "deprel_g" / "deprel_p" etc. when
                       comparing gold vs predicted within STR.
    ud_label_col     : analogous for UD.
    extra_str_cols   : extra columns from `str_df` to carry over as
                       `str_<col>` in the output (e.g. head_p for error
                       matrices).
    extra_ud_cols    : analogous for UD (prefix `ud_`).
    """
    # Fast lookup tables indexed by (sent_id, id)
    str_idx = _indexed(str_df)
    ud_idx = _indexed(ud_df)

    rows = []
    for sent_id, sc in corr.per_sentence.items():
        ud_sk = corr.ud_skel[sent_id]
        str_sk = corr.str_skel.get(sent_id)

        for e_ud, m in sc.matches.items():
            dep_ud = ud_sk.dep_of(e_ud)
            head_ud = ud_sk.head_of(e_ud)
            if dep_ud is None:
                continue

            ud_row = ud_idx.get((sent_id, dep_ud))
            if ud_row is None:
                continue
            if exclude_punct and _is_punct(ud_row, ud_label_col):
                continue

            status = m.status
            str_token: Optional[TokenID] = None
            str_head: Optional[TokenID] = None
            if m.e_str is not None and str_sk is not None:
                str_token = str_sk.dep_of(m.e_str)
                str_head = str_sk.head_of(m.e_str)

            str_row = str_idx.get((sent_id, str_token)) if str_token is not None else None

            record = {
                "sent_id": sent_id,
                "ud_id": dep_ud,
                "ud_head": head_ud,
                "ud_form": ud_row.get("form"),
                "ud_deprel": ud_row.get(ud_label_col),
                "str_id": str_token,
                "str_head": str_head,
                "str_form": (str_row or {}).get("form"),
                "str_deprel": (str_row or {}).get(str_label_col),
                "status": status,
                "comparable": status in ("exact_same_dir", "exact_mirrored", "restructured"),
            }
            for col in extra_ud_cols:
                record[f"ud_{col}"] = ud_row.get(col)
            for col in extra_str_cols:
                record[f"str_{col}"] = (str_row or {}).get(col)
            rows.append(record)

    cols = [
        "sent_id", "ud_id", "ud_form", "ud_head", "ud_deprel",
        "str_id", "str_form", "str_head", "str_deprel",
        "status", "comparable",
    ] + [f"ud_{c}" for c in extra_ud_cols] + [f"str_{c}" for c in extra_str_cols]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _indexed(df: pd.DataFrame) -> dict:
    """
    Build a (sent_id, id) → row-as-dict index for O(1) lookup.

    We deliberately materialise the rows as dicts rather than rely on
    .loc[(sid, tid)] because the latter copies the row on every lookup
    and is noticeably slower for corpus-sized inputs.
    """
    cols = [c for c in df.columns if c not in ("sent_id", "id")]
    sent = df["sent_id"].to_list()
    ids = df["id"].to_numpy()
    data = {c: df[c].to_list() for c in cols}
    out = {}
    for i, (sid, tid) in enumerate(zip(sent, ids)):
        row = {c: data[c][i] for c in cols}
        out[(sid, int(tid))] = row
    return out


def _is_punct(ud_row: dict, label_col: str) -> bool:
    deprel = ud_row.get(label_col)
    if isinstance(deprel, str) and deprel.lower() == "punct":
        return True
    upos = ud_row.get("upos")
    if isinstance(upos, str) and upos.upper() == "PUNCT":
        return True
    return False


# ---------------------------------------------------------------------------
# Convenience aggregates
# ---------------------------------------------------------------------------

def label_confusion(
    table: pd.DataFrame,
    *,
    status: tuple[str, ...] = ("exact_same_dir", "exact_mirrored", "restructured"),
) -> pd.DataFrame:
    """
    Confusion table str_deprel × ud_deprel, restricted to comparable rows.

    Rows where `comparable` is False are silently excluded. Returns a
    long DataFrame with columns [str_deprel, ud_deprel, count] sorted
    by count descending. Use pivot_table downstream if a matrix form is
    desired.
    """
    sub = table[table["status"].isin(status)]
    grouped = (
        sub.groupby(["str_deprel", "ud_deprel"], dropna=False)
           .size()
           .reset_index(name="count")
           .sort_values("count", ascending=False)
           .reset_index(drop=True)
    )
    return grouped


def coverage_by_mode(corr: EdgeCorrespondence) -> pd.DataFrame:
    """
    One-row summary of an EdgeCorrespondence: mode, counts per status,
    resolved percentage, baseline percentage.
    """
    stats = corr.coverage()
    by = stats["by_status"]
    return pd.DataFrame([{
        "mode": stats["mode"],
        "total": stats["total_ud_edges"],
        "exact_same_dir": by["exact_same_dir"],
        "exact_mirrored": by["exact_mirrored"],
        "restructured": by["restructured"],
        "unresolved": by["unresolved"],
        "resolved_pct": round(stats["resolved_pct"], 2),
        "baseline_pct": round(stats["baseline_pct"], 2),
    }])