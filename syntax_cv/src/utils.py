"""
utils.py — build paired (gold, predicted) dataframes from CV conllu files
and attach token-level label comparisons driven by an EdgeCorrespondence.

Rewritten to remove all tree-mutating steps. The previous version of this
module called `apply_swap_plan` / `apply_token_mapping` to rewrite the STR
DataFrame so that its row indices aligned with UD before comparison. The
supervisor's feedback (16-Apr-26) is that the two gold trees must remain
untouched; the comparison is a lookup, not a rewrite.

Public entry points:

    build_data(...)            — load (gold, pred) CV pairs for both corpora.
    build_correspondence(...)  — build strict and/or extended correspondences
                                 from the gold tree pair.
    attach_comparisons(...)    — annotate every row with the status produced
                                 by the chosen correspondence and the
                                 counterpart STR token id, without
                                 altering STR/UD indices.
    filter_consistent(...)     — kept as-is (filters tokens on which
                                 all model runs agree).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from src.convert import convert_cv_to_csv

from edge_correspondence import (
    EdgeCorrespondence,
    build_edge_correspondence,
)
from lca_candidates import build_lca_candidates
from token_comparison import build_comparison_table


# ---------------------------------------------------------------------------
# CV loading (unchanged behaviour, now consuming pair csvs as before)
# ---------------------------------------------------------------------------

def load_pair(gold_csv, pred_csv) -> pd.DataFrame:
    """
    Merge one gold and one predicted CSV on (sent_id, id).

    Output columns: text, form, upos, feats, head_g, head_p, deprel_g,
    deprel_p, ellipsis.
    Index: (sent_id, id).
    """
    g = pd.read_csv(gold_csv, dtype={"sent_id": str})
    p = pd.read_csv(pred_csv, dtype={"sent_id": str})

    if g["id"].dtype != "int64":
        g["id"] = g["id"].astype("int64", errors="ignore")
    if p["id"].dtype != "int64":
        p["id"] = p["id"].astype("int64", errors="ignore")

    merged = g.merge(p, on=["sent_id", "id"], how="inner", suffixes=("_g", "_p"))

    for name in ("text", "form", "upos", "feats", "ellipsis"):
        if f"{name}_p" in merged.columns:
            merged = merged.drop(columns=[f"{name}_p"])
        if f"{name}_g" in merged.columns:
            merged = merged.rename(columns={f"{name}_g": name})

    merged = merged.set_index(["sent_id", "id"]).sort_index()
    cols = ["text", "form", "upos", "feats",
            "head_g", "head_p", "deprel_g", "deprel_p", "ellipsis"]
    return merged[[c for c in cols if c in merged.columns]]


def build_data(
    out_root: str = "../out/UDPipe2/mmBERT",
    csv_dir: str = "../syntax_cv/data/cv",
    corpora: Optional[list[str]] = None,
    num_runs: int = 5,
    *,
    force: bool = False,
) -> dict:
    """
    Convert merged CV conllu files to CSV and load (gold, pred) pairs.

    NOTE: this function no longer aligns STR against UD. The previous
    `aligned=True` / `align_mode` code path has been removed because it
    mutated the STR tree; downstream code is expected to consume the
    pair tables and an EdgeCorrespondence side-by-side.

    Returns:
      {
        "str-new": {1: df_run1, ..., N: df_runN},
        "ud-new":  {1: df_run1, ...},
      }
    """
    if corpora is None:
        corpora = ["str-new", "ud-new"]

    index = convert_cv_to_csv(
        out_root=out_root,
        csv_dir=csv_dir,
        corpora=corpora,
        num_runs=num_runs,
        force=force,
    )

    data: dict = {}
    for corpus, paths in index.items():
        gold_csv = paths["gold"]
        if gold_csv is None or not Path(gold_csv).exists():
            print(f"[warn] no gold CSV for {corpus}, skipping")
            continue

        runs: dict = {}
        for run_idx, pred_csv in paths["pred"].items():
            if not Path(pred_csv).exists():
                print(f"[warn] no pred CSV for {corpus} run{run_idx}, skipping")
                continue
            runs[run_idx] = load_pair(gold_csv, pred_csv)

        data[corpus] = runs

    return data


# ---------------------------------------------------------------------------
# Correspondence building
# ---------------------------------------------------------------------------

def build_correspondence(
    str_gold: pd.DataFrame,
    ud_gold: pd.DataFrame,
    *,
    mode: Literal["strict", "extended"] = "strict",
    max_path_len: int = 3,
) -> EdgeCorrespondence:
    """
    Build a single EdgeCorrespondence from two gold-tree DataFrames.

    The inputs are expected to include columns sent_id, id, head (and
    ideally form, upos, deprel for downstream analysis). Both can be
    either index-reset or (sent_id, id)-indexed.

    In `extended` mode, LCA-derived candidates are generated with
    `build_lca_candidates(max_path_len=max_path_len)` and injected
    into the CP resolver; no modification is performed on the inputs.
    """
    str_df = str_gold.reset_index() if _is_mi_indexed(str_gold) else str_gold
    ud_df = ud_gold.reset_index() if _is_mi_indexed(ud_gold) else ud_gold

    # Normalize head column name: CV pairs carry head_g + head_p; the
    # correspondence consumes the gold tree, so rename on the fly.
    str_df = _as_gold_view(str_df)
    ud_df = _as_gold_view(ud_df)

    extras = None
    if mode == "extended":
        extras = build_lca_candidates(str_df, ud_df, max_path_len=max_path_len)

    return build_edge_correspondence(
        str_df, ud_df, mode=mode, extra_candidates=extras,
    )


def _is_mi_indexed(df: pd.DataFrame) -> bool:
    return isinstance(df.index, pd.MultiIndex) or df.index.name in ("sent_id", "id")


def _as_gold_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a shallow view with a single `head` column derived from
    `head_g` (preferred) or `head` (if already there). Same for `deprel`.
    Never mutates the source frame.
    """
    cols = {}
    cols["sent_id"] = df["sent_id"]
    cols["id"] = df["id"]
    if "head_g" in df.columns:
        cols["head"] = df["head_g"]
    elif "head" in df.columns:
        cols["head"] = df["head"]
    else:
        raise KeyError("no head / head_g column")
    if "deprel_g" in df.columns:
        cols["deprel"] = df["deprel_g"]
    elif "deprel" in df.columns:
        cols["deprel"] = df["deprel"]
    for extra in ("form", "upos", "feats"):
        if extra in df.columns:
            cols[extra] = df[extra]
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def attach_comparisons(
    str_gold: pd.DataFrame,
    ud_gold: pd.DataFrame,
    corr: EdgeCorrespondence,
    *,
    exclude_punct: bool = True,
) -> pd.DataFrame:
    """
    Produce a per-token comparison table from gold trees and a
    correspondence. Delegates to `token_comparison.build_comparison_table`.
    """
    str_df = _as_gold_view(
        str_gold.reset_index() if _is_mi_indexed(str_gold) else str_gold
    )
    ud_df = _as_gold_view(
        ud_gold.reset_index() if _is_mi_indexed(ud_gold) else ud_gold
    )
    return build_comparison_table(
        str_df, ud_df, corr, exclude_punct=exclude_punct,
    )


# ---------------------------------------------------------------------------
# Consistent-token filter (kept verbatim from previous version)
# ---------------------------------------------------------------------------

def filter_consistent(data: dict) -> dict:
    """
    Keep only tokens where all model runs agree on deprel_p and head_p.
    The same token mask is applied to STR and UD, so the two datasets stay
    aligned on the same (sent_id, id) index.

    Input:  data from build_data → {corpus: {run_idx: df}}
    Output: {corpus: df}
    """
    ud_runs = data.get("ud-new", {})
    str_runs = data.get("str-new", {})
    if not ud_runs or not str_runs:
        raise ValueError("str-new or ud-new not found in data.")

    def _consistent_mask(runs):
        deprel = pd.concat([df["deprel_p"].rename(i) for i, df in runs.items()], axis=1)
        head = pd.concat([df["head_p"].rename(i) for i, df in runs.items()], axis=1)
        return (deprel.nunique(axis=1) == 1) & (head.nunique(axis=1) == 1)

    ud_mask = _consistent_mask(ud_runs)
    str_mask = _consistent_mask(str_runs)
    consistent_mask = ud_mask & str_mask

    n_total = len(consistent_mask)
    print(f"[filter] str-new consistent: {str_mask.sum()}/{n_total} "
          f"({100*str_mask.mean():.1f}%)")
    print(f"[filter] ud-new  consistent: {ud_mask.sum()}/{n_total} "
          f"({100*ud_mask.mean():.1f}%)")
    print(f"[filter] both    consistent: {consistent_mask.sum()}/{n_total} "
          f"({100*consistent_mask.mean():.1f}%)")

    result = {}
    for corpus, runs in data.items():
        if not runs:
            continue
        first_df = next(iter(runs.values()))
        result[corpus] = first_df[consistent_mask]

    return result