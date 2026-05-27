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

import re
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from src.convert import convert_cv_to_csv, read_conllu

from src.edge_correspondence import (
    EdgeCorrespondence,
    build_edge_correspondence,
)
from src.lca_candidates import build_lca_candidates
from src.token_comparison import build_comparison_table
from src.token_comparison import coverage_by_mode, label_confusion


_SENT_ID_RE = re.compile(r"^#\s*sent_id\s*=\s*(.*)$", flags=re.IGNORECASE)
_TEXT_RE = re.compile(r"^#\s*text\s*=\s*(.*)$", flags=re.IGNORECASE)


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
# Full-corpus aligned loading (Aligned/*.conllu)
# ---------------------------------------------------------------------------

def _extract_comment_value(comments: list[str], regex: re.Pattern[str]) -> Optional[str]:
    for c in comments:
        m = regex.match(c)
        if m:
            return m.group(1).strip()
    return None


def conllu_to_gold_df(conllu_path: str | Path) -> pd.DataFrame:
    """
    Read one CoNLL-U file into a gold-tree DataFrame.

    Output columns:
      sent_id, text, id, form, upos, feats, head, deprel
    """
    rows: list[dict[str, object]] = []
    for sent_idx, sent in enumerate(read_conllu(str(conllu_path)), start=1):
        sent_id = _extract_comment_value(sent.comments, _SENT_ID_RE)
        text = _extract_comment_value(sent.comments, _TEXT_RE)
        if sent_id is None:
            sent_id = f"sent_{sent_idx}"

        for tok in sent.tokens:
            try:
                head = int(tok.HEAD)
            except (TypeError, ValueError):
                head = 0
            rows.append(
                {
                    "sent_id": sent_id,
                    "text": text,
                    "id": int(tok.ID),
                    "form": tok.FORM,
                    "upos": "" if tok.UPOS in (None, "_") else tok.UPOS,
                    "feats": "" if tok.FEATS in (None, "_") else tok.FEATS,
                    "head": head,
                    "deprel": "" if tok.DEPREL in (None, "_") else tok.DEPREL,
                }
            )

    out = pd.DataFrame(
        rows,
        columns=["sent_id", "text", "id", "form", "upos", "feats", "head", "deprel"],
    )
    if len(out):
        out["id"] = pd.to_numeric(out["id"], errors="raise").astype("int64")
        out["head"] = pd.to_numeric(out["head"], errors="raise").astype("int64")
    return out


def summarize_parallel_tokenization(
    str_gold: pd.DataFrame,
    ud_gold: pd.DataFrame,
) -> dict[str, int]:
    """
    Summary diagnostics for paired tokenization consistency.

    The aligned full corpus is expected to be 1:1 on (sent_id, id, form).
    """
    str_df = _as_gold_view(
        str_gold.reset_index() if _is_mi_indexed(str_gold) else str_gold
    )
    ud_df = _as_gold_view(
        ud_gold.reset_index() if _is_mi_indexed(ud_gold) else ud_gold
    )

    str_sent = set(str_df["sent_id"].dropna().astype(str))
    ud_sent = set(ud_df["sent_id"].dropna().astype(str))

    str_counts = str_df.groupby("sent_id", as_index=False).size().rename(columns={"size": "n_str"})
    ud_counts = ud_df.groupby("sent_id", as_index=False).size().rename(columns={"size": "n_ud"})
    cnt = str_counts.merge(ud_counts, on="sent_id", how="outer")
    count_mismatch = int((cnt["n_str"] != cnt["n_ud"]).fillna(True).sum())

    pair_cmp = ud_df[["sent_id", "id", "form"]].merge(
        str_df[["sent_id", "id", "form"]],
        on=["sent_id", "id"],
        how="outer",
        suffixes=("_ud", "_str"),
        indicator=True,
    )
    missing_pairs = int((pair_cmp["_merge"] != "both").sum())
    form_mismatches = int(
        ((pair_cmp["_merge"] == "both") & (pair_cmp["form_ud"] != pair_cmp["form_str"])).sum()
    )

    return {
        "sentences_str": int(str_df["sent_id"].nunique()),
        "sentences_ud": int(ud_df["sent_id"].nunique()),
        "sent_id_only_in_str": int(len(str_sent - ud_sent)),
        "sent_id_only_in_ud": int(len(ud_sent - str_sent)),
        "token_rows_str": int(len(str_df)),
        "token_rows_ud": int(len(ud_df)),
        "sentence_count_mismatches": count_mismatch,
        "missing_sent_id_id_pairs": missing_pairs,
        "form_mismatches_same_sent_id_id": form_mismatches,
    }


def load_aligned_corpus_gold(
    *,
    aligned_dir: str = "../Aligned",
    str_file: str = "str_aligned.conllu",
    ud_file: str = "ud_aligned.conllu",
    strict: bool = True,
    return_summary: bool = False,
):
    """
    Load the full aligned corpus pair from `Aligned/*.conllu`.

    Parameters
    ----------
    strict
        If True, require perfect 1:1 tokenization agreement between STR/UD
        on (sent_id, id, form); otherwise only load and report summary.
    return_summary
        If True, return `(str_gold, ud_gold, summary_dict)`.
    """
    root = Path(aligned_dir)
    str_path = root / str_file
    ud_path = root / ud_file

    if not str_path.exists():
        raise FileNotFoundError(f"STR aligned file not found: {str_path}")
    if not ud_path.exists():
        raise FileNotFoundError(f"UD aligned file not found: {ud_path}")

    str_gold = conllu_to_gold_df(str_path)
    ud_gold = conllu_to_gold_df(ud_path)
    summary = summarize_parallel_tokenization(str_gold, ud_gold)

    if strict:
        bad = {
            k: v
            for k, v in summary.items()
            if k in {
                "sent_id_only_in_str",
                "sent_id_only_in_ud",
                "sentence_count_mismatches",
                "missing_sent_id_id_pairs",
                "form_mismatches_same_sent_id_id",
            }
            and v != 0
        }
        if bad:
            raise ValueError(
                "Aligned corpus is not 1:1 token-parallel. "
                f"Mismatches: {bad}"
            )

    if return_summary:
        return str_gold, ud_gold, summary
    return str_gold, ud_gold


# ---------------------------------------------------------------------------
# Correspondence building
# ---------------------------------------------------------------------------

def build_correspondence(
    str_gold: pd.DataFrame,
    ud_gold: pd.DataFrame,
    *,
    mode: Literal[
        "strict",
        "strict_plus",
        "extended",
        "extended_stepwise",
        "topdown",
        "fi2003",
        "certified",
    ] = "strict",
    max_path_len: int = 3,
    fi_k: Optional[int] = 8,
    certified_use_fi2003: bool = False,
    certified_use_punct_band: bool = True,
) -> EdgeCorrespondence:
    """
    Build a single EdgeCorrespondence from two gold-tree DataFrames.

    The inputs are expected to include columns sent_id, id, head (and
    ideally form, upos, deprel for downstream analysis). Both can be
    either index-reset or (sent_id, id)-indexed.

    In `strict_plus` mode, a local 2-hop bridge heuristic is enabled
    inside `build_edge_correspondence`.

    In `extended` mode, LCA-derived candidates are generated with
    `build_lca_candidates(max_path_len=max_path_len)` and injected
    into the CP resolver; no modification is performed on the inputs.

    In `extended_stepwise` mode, we run iterative extended passes with
    cumulative candidate pools:
      strict -> LCA<=2 -> LCA<=3 -> ... -> LCA<=max_path_len.

    In `topdown` mode, trees are matched top-down from the roots:
    exact edges, shared children, singleton elimination, and brute-force
    overlap scoring.  No LCA candidates or extra_candidates are used.

    In `fi2003` mode, ordered-tree alignment DP (FI2003/JWZ style) is used.
    `fi_k` controls k-relevance filtering. The default (`fi_k=8`) is a
    practical speed/coverage trade-off for sentence-length inputs. Set
    `fi_k=None` to enable automatic doubling with an unfiltered fallback.

    In `certified` mode, multiple candidate generators are combined into a
    single candidate pool, followed by a two-stage certification:
      1. strict structural forcing;
      2. secondary heuristic disambiguation.
    Residual non-unique cases are marked as `ambiguous`, and candidate
    failures are marked as `candidate_gap`.
    Set `certified_use_fi2003=True` to add FI2003 as an auxiliary candidate
    generator; it is disabled by default because it is substantially slower
    and was not the strongest standalone method in prior experiments.
    `certified_use_punct_band` is kept only for backward compatibility and
    is ignored: punctuation-specific residual heuristics have been removed.
    """
    str_df = str_gold.reset_index() if _is_mi_indexed(str_gold) else str_gold
    ud_df = ud_gold.reset_index() if _is_mi_indexed(ud_gold) else ud_gold

    # Normalize head column name: CV pairs carry head_g + head_p; the
    # correspondence consumes the gold tree, so rename on the fly.
    str_df = _as_gold_view(str_df)
    ud_df = _as_gold_view(ud_df)

    extras = None
    extras_steps = None
    if mode == "extended":
        extras = build_lca_candidates(str_df, ud_df, max_path_len=max_path_len)
    elif mode == "extended_stepwise":
        start = 2
        if max_path_len < start:
            extras_steps = []
        else:
            extras_steps = [
                build_lca_candidates(str_df, ud_df, max_path_len=d)
                for d in range(start, max_path_len + 1)
            ]
    elif mode == "certified":
        start = 2
        if max_path_len < start:
            extras_steps = []
        else:
            extras_steps = [
                build_lca_candidates(str_df, ud_df, max_path_len=d)
                for d in range(start, max_path_len + 1)
            ]
        extras = build_lca_candidates(str_df, ud_df, max_path_len=None)

    return build_edge_correspondence(
        str_df,
        ud_df,
        mode=mode,
        extra_candidates=extras,
        extra_candidates_steps=extras_steps,
        fi_k=fi_k,
        certified_use_fi2003=certified_use_fi2003,
        certified_use_punct_band=certified_use_punct_band,
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


def export_alignment_csvs(
    str_gold: pd.DataFrame,
    ud_gold: pd.DataFrame,
    corr: EdgeCorrespondence,
    *,
    mode_name: Optional[str] = None,
    out_dir: str = "syntax_cv/data/aligned",
    exclude_punct: bool = True,
) -> dict[str, Path]:
    """
    Save alignment outputs into `data/aligned/`.

    Files written:
      - compare_<mode>.csv                (historical 11-column schema)
      - compare_<mode>_detailed.csv       (status+certification diagnostics)
      - label_confusion_<mode>.csv
      - coverage_<mode>.csv

    Returns a dict name -> Path for downstream logging.
    """
    mode = mode_name or corr.mode
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    str_df = _as_gold_view(
        str_gold.reset_index() if _is_mi_indexed(str_gold) else str_gold
    )
    ud_df = _as_gold_view(
        ud_gold.reset_index() if _is_mi_indexed(ud_gold) else ud_gold
    )

    table_full = build_comparison_table(
        str_df,
        ud_df,
        corr,
        exclude_punct=exclude_punct,
    )
    base_cols = [
        "sent_id", "ud_id", "ud_form", "ud_head", "ud_deprel",
        "str_id", "str_form", "str_head", "str_deprel",
        "status", "comparable",
    ]
    table_base = table_full[base_cols].copy()

    p_compare = out / f"compare_{mode}.csv"
    table_base.to_csv(p_compare, index=False, encoding="utf-8")

    detail_rows: list[dict] = []
    for sent_id, sc in corr.per_sentence.items():
        ud_sk = corr.ud_skel[sent_id]
        str_sk = corr.str_skel.get(sent_id)
        for e_ud, m in sc.matches.items():
            dep_ud = ud_sk.dep_of(e_ud)
            head_ud = ud_sk.head_of(e_ud)
            if dep_ud is None:
                continue
            dep_str = str_sk.dep_of(m.e_str) if (str_sk is not None and m.e_str is not None) else None
            head_str = str_sk.head_of(m.e_str) if (str_sk is not None and m.e_str is not None) else None
            detail_rows.append(
                {
                    "sent_id": sent_id,
                    "ud_id": dep_ud,
                    "ud_head": head_ud,
                    "str_id": dep_str,
                    "str_head": head_str,
                    "status": m.status,
                    "certification": getattr(m, "certification", "strict"),
                    "detail": getattr(m, "detail", ""),
                    "candidate_count": getattr(m, "candidate_count", 0),
                    "support_count": getattr(m, "support_count", 0),
                }
            )
    p_detail = out / f"compare_{mode}_detailed.csv"
    pd.DataFrame(detail_rows).to_csv(p_detail, index=False, encoding="utf-8")

    p_conf = out / f"label_confusion_{mode}.csv"
    label_confusion(table_base).to_csv(p_conf, index=False, encoding="utf-8")

    p_cov = out / f"coverage_{mode}.csv"
    coverage_by_mode(corr, include_punct=not exclude_punct).to_csv(
        p_cov, index=False, encoding="utf-8"
    )

    return {
        "compare": p_compare,
        "detailed": p_detail,
        "label_confusion": p_conf,
        "coverage": p_cov,
    }


def _build_alignment_maps(
    corr: EdgeCorrespondence,
    *,
    aligned_col: str = "aligned_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build UD->STR and STR->UD token-id maps from an edge correspondence.
    """

    def _roots(sk) -> set[int]:
        deps = {d for d in (sk.dep_of(e) for e in sk.edges) if d is not None}
        return {int(v) for v in sk.token_ids if int(v) not in deps}

    ud_to_str: dict[str, dict[int, int]] = {}
    for sent_id, ud_sk in corr.ud_skel.items():
        local: dict[int, int] = {}
        for tid in ud_sk.token_ids:
            t = int(tid)
            s = corr.counterpart_str_token(sent_id, t)
            if s is not None:
                local[t] = int(s)

        str_sk = corr.str_skel.get(sent_id)
        if str_sk is not None:
            ud_roots = sorted(_roots(ud_sk))
            str_roots = sorted(_roots(str_sk))
            if len(ud_roots) == 1 and len(str_roots) == 1:
                local.setdefault(ud_roots[0], str_roots[0])
            else:
                for r in set(ud_roots) & set(str_roots):
                    local.setdefault(int(r), int(r))

        ud_to_str[str(sent_id)] = local

    str_to_ud: dict[str, dict[int, int]] = {}
    for sent_id, mapping in ud_to_str.items():
        inv: dict[int, int | None] = {}
        for ud_id, str_id in mapping.items():
            if str_id in inv and inv[str_id] != ud_id:
                inv[str_id] = None
            else:
                inv[str_id] = ud_id
        str_to_ud[sent_id] = {s: u for s, u in inv.items() if u is not None}

    ud_records: list[dict[str, object]] = []
    for sent_id, m in ud_to_str.items():
        for ud_id, str_id in m.items():
            ud_records.append({"sent_id": sent_id, "id": int(ud_id), aligned_col: int(str_id)})
    ud_map = pd.DataFrame(ud_records, columns=["sent_id", "id", aligned_col])

    str_records: list[dict[str, object]] = []
    for sent_id, m in str_to_ud.items():
        for str_id, ud_id in m.items():
            str_records.append({"sent_id": sent_id, "id": int(str_id), aligned_col: int(ud_id)})
    str_map = pd.DataFrame(str_records, columns=["sent_id", "id", aligned_col])

    return ud_map, str_map


def export_aligned_pair_csvs(
    str_gold: pd.DataFrame,
    ud_gold: pd.DataFrame,
    corr: EdgeCorrespondence,
    *,
    out_dir: str = "syntax_cv/data/aligned_full",
    str_name: str = "str-aligned.csv",
    ud_name: str = "ud-aligned.csv",
    aligned_col: str = "aligned_id",
) -> dict[str, Path]:
    """
    Export two aligned tables (STR and UD) with one extra `aligned_id` column.

    This is intended for full-corpus workflows (e.g. `Aligned/*.conllu`).
    """
    str_src = str_gold.reset_index() if _is_mi_indexed(str_gold) else str_gold.copy()
    ud_src = ud_gold.reset_index() if _is_mi_indexed(ud_gold) else ud_gold.copy()
    str_src["sent_id"] = str_src["sent_id"].astype(str)
    ud_src["sent_id"] = ud_src["sent_id"].astype(str)

    ud_map, str_map = _build_alignment_maps(corr, aligned_col=aligned_col)

    ud_out = ud_src.merge(ud_map, on=["sent_id", "id"], how="left")
    str_out = str_src.merge(str_map, on=["sent_id", "id"], how="left")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ud_path = out_root / ud_name
    str_path = out_root / str_name
    ud_out.to_csv(ud_path, index=False, encoding="utf-8")
    str_out.to_csv(str_path, index=False, encoding="utf-8")

    return {"ud": ud_path, "str": str_path}


def export_aligned_cv_csvs(
    str_gold: pd.DataFrame,
    ud_gold: pd.DataFrame,
    corr: EdgeCorrespondence,
    *,
    cv_dir: str = "syntax_cv/data/cv",
    out_dir: str = "syntax_cv/data/aligned",
    aligned_col: str = "aligned_id",
) -> dict[str, Path]:
    """
    Export alignment in a `data/cv`-like format (same files + `aligned_id`).
    """
    _ = str_gold
    _ = ud_gold
    ud_map, str_map = _build_alignment_maps(corr, aligned_col=aligned_col)

    cv_root = Path(cv_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for src in sorted(cv_root.glob("*.csv")):
        name = src.name
        df = pd.read_csv(src, dtype={"sent_id": str})
        if "sent_id" not in df.columns or "id" not in df.columns:
            continue

        if name.startswith("ud-"):
            map_df = ud_map
        elif name.startswith("str-"):
            map_df = str_map
        else:
            continue

        out_df = df.merge(map_df, on=["sent_id", "id"], how="left")
        dst = out_root / name
        out_df.to_csv(dst, index=False, encoding="utf-8")
        written[name] = dst

    return written


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
