import pandas as pd
from pathlib import Path

from src.convert import convert_cv_to_csv
from src.align_markup import (
    apply_swap_plan,
    build_recursive_swap_plan_cp,
)


def load_pair(gold_csv, pred_csv) -> pd.DataFrame:
    g = pd.read_csv(gold_csv, dtype={"sent_id": str})
    p = pd.read_csv(pred_csv, dtype={"sent_id": str})

    if g["id"].dtype != "int64":
        g["id"] = g["id"].astype("int64", errors="ignore")
    if p["id"].dtype != "int64":
        p["id"] = p["id"].astype("int64", errors="ignore")

    merged = g.merge(p, on=["sent_id", "id"], how="inner", suffixes=("_g", "_p"))

    if "text_p" in merged.columns:
        merged = merged.drop(columns=["text_p"])
    merged = merged.rename(columns={"text_g": "text"})

    if "form_p" in merged.columns:
        merged = merged.drop(columns=["form_p"])
    merged = merged.rename(columns={"form_g": "form"})

    if "upos_p" in merged.columns:
        merged = merged.drop(columns=["upos_p"])
    merged = merged.rename(columns={"upos_g": "upos"})

    if "feats_p" in merged.columns:
        merged = merged.drop(columns=["feats_p"])
    merged = merged.rename(columns={"feats_g": "feats"})

    if "ellipsis_p" in merged.columns:
        merged = merged.drop(columns=["ellipsis_p"])
    if "ellipsis_g" in merged.columns:
        merged = merged.rename(columns={"ellipsis_g": "ellipsis"})

    merged = merged.set_index(["sent_id", "id"]).sort_index()

    cols = ["text", "form", "upos", "feats", "head_g", "head_p", "deprel_g", "deprel_p", "ellipsis"]
    return merged[cols]


def _align_cv(index: dict, aligned_dir: Path, num_runs: int, force: bool) -> dict:
    """
    Align str-new CSVs against ud-new CSVs and save to aligned_dir.

    Swap pairs are derived from gold via CP (same for all runs).
    Swaps are applied to str-new gold and each str-new pred run.
    ud-new files are copied as-is.

    Returns the same index structure but pointing to aligned_dir.
    """
    aligned_dir.mkdir(parents=True, exist_ok=True)

    str_paths = index.get("str-new", {})
    ud_paths  = index.get("ud-new",  {})

    str_gold_csv = str_paths.get("gold")
    ud_gold_csv  = ud_paths.get("gold")

    if str_gold_csv is None or ud_gold_csv is None:
        raise FileNotFoundError("str-new or ud-new gold CSV not found; run without aligned first.")

    # Check if all output files already exist — skip alignment entirely if so
    expected_outputs = [
        aligned_dir / Path(str_gold_csv).name,
        aligned_dir / Path(ud_gold_csv).name,
    ] + [
        aligned_dir / Path(p).name
        for d in (str_paths.get("pred", {}), ud_paths.get("pred", {}))
        for p in d.values()
        if p and Path(p).exists()
    ]

    if not force and all(p.exists() for p in expected_outputs):
        print("[align] all aligned files already exist, skipping alignment")
        aligned_index: dict = {
            "str-new": {"gold": expected_outputs[0], "pred": {}},
            "ud-new":  {"gold": expected_outputs[1], "pred": {}},
        }
        for run_idx, p in str_paths.get("pred", {}).items():
            if p:
                aligned_index["str-new"]["pred"][run_idx] = aligned_dir / Path(p).name
        for run_idx, p in ud_paths.get("pred", {}).items():
            if p:
                aligned_index["ud-new"]["pred"][run_idx] = aligned_dir / Path(p).name
        return aligned_index

    str_gold_df = pd.read_csv(str_gold_csv, dtype={"sent_id": str})
    ud_gold_df  = pd.read_csv(ud_gold_csv,  dtype={"sent_id": str})

    # CP recursive swap plan computed on gold and reused for all runs
    plan, plan_diag = build_recursive_swap_plan_cp(
        str_gold_df,
        ud_gold_df,
        max_passes=20,
        stop_on_cycle=True,
        return_diagnostics=True,
    )
    n_pairs = sum(len(pass_pairs) for pass_pairs in plan)
    if n_pairs:
        print(f"[align] CP recursive swap: {n_pairs} pairs in {len(plan)} passes")
    else:
        print("[align] CP recursive swap: no pairs found")
    if plan_diag.get("any_unresolved"):
        unresolved = plan_diag.get("last_unresolved_ud_edges", 0)
        unresolved_sents = plan_diag.get("last_sentences_with_unresolved", 0)
        print(
            "[align][warn] CP unresolved edges remain after the last pass: "
            f"{unresolved} edges in {unresolved_sents} sentences"
        )

    str_gold_aligned = apply_swap_plan(str_gold_df, plan)

    aligned_index: dict = {"str-new": {}, "ud-new": {}}

    str_gold_out = aligned_dir / Path(str_gold_csv).name
    ud_gold_out  = aligned_dir / Path(ud_gold_csv).name

    if not str_gold_out.exists() or force:
        str_gold_aligned.to_csv(str_gold_out, index=False, encoding="utf-8")
        print(f"[align] saved {str_gold_out}")

    if not ud_gold_out.exists() or force:
        ud_gold_df.to_csv(ud_gold_out, index=False, encoding="utf-8")
        print(f"[align] saved {ud_gold_out}")

    aligned_index["str-new"]["gold"] = str_gold_out
    aligned_index["ud-new"]["gold"]  = ud_gold_out
    aligned_index["str-new"]["pred"] = {}
    aligned_index["ud-new"]["pred"]  = {}

    for run_idx in range(1, num_runs + 1):
        str_pred_csv = str_paths.get("pred", {}).get(run_idx)
        ud_pred_csv  = ud_paths.get("pred", {}).get(run_idx)

        if str_pred_csv and Path(str_pred_csv).exists():
            str_pred_df = pd.read_csv(str_pred_csv, dtype={"sent_id": str})
            str_pred_df = apply_swap_plan(str_pred_df, plan)
            str_pred_out = aligned_dir / Path(str_pred_csv).name
            if not str_pred_out.exists() or force:
                str_pred_df.to_csv(str_pred_out, index=False, encoding="utf-8")
            aligned_index["str-new"]["pred"][run_idx] = str_pred_out

        if ud_pred_csv and Path(ud_pred_csv).exists():
            ud_pred_df = pd.read_csv(ud_pred_csv, dtype={"sent_id": str})
            ud_pred_out = aligned_dir / Path(ud_pred_csv).name
            if not ud_pred_out.exists() or force:
                ud_pred_df.to_csv(ud_pred_out, index=False, encoding="utf-8")
            aligned_index["ud-new"]["pred"][run_idx] = ud_pred_out

    return aligned_index


def build_data(
    out_root: str = "../out/UDPipe2/mmBERT",
    csv_dir: str = "../syntax_cv/data/cv",
    corpora: list[str] = None,
    num_runs: int = 5,
    *,
    aligned: bool = False,
    force: bool = False,
) -> dict:
    """
    Convert merged CV conllu files to CSV and load (gold, pred) pairs.

    If aligned=True, aligns str-new markup against ud-new before loading
    and saves aligned CSVs to syntax_cv/data/cv_aligned/.

    Returns:
      {
        "str-new": {1: df_run1, ..., 5: df_run5},
        "ud-new":  {1: df_run1, ...},
      }

    Each df has columns: text, form, upos, feats,
                         head_g, head_p, deprel_g, deprel_p, ellipsis
    indexed by (sent_id, id).
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

    if aligned:
        aligned_dir = Path(csv_dir).parent / "cv_aligned"
        index = _align_cv(index, aligned_dir, num_runs, force)

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


def filter_consistent(data: dict) -> dict:
    """
    Keep only tokens where all UD model runs predicted the same deprel_p and head_p.
    The same token mask is applied to STR, so both datasets stay aligned.

    Input:  data from build_data → {corpus: {run_idx: df}}
    Output: {corpus: df}  — same format as build_data_from_suffix in syntax/src/utils.py

    Each output df is indexed by (sent_id, id) and contains the same columns
    as a single-run df (text, form, upos, feats, head_g, head_p, deprel_g, deprel_p, ellipsis).
    """
    ud_runs  = data.get("ud-new",  {})
    str_runs = data.get("str-new", {})
    if not ud_runs or not str_runs:
        raise ValueError("str-new or ud-new not found in data.")

    def _consistent_mask(runs):
        deprel = pd.concat([df["deprel_p"].rename(i) for i, df in runs.items()], axis=1)
        head   = pd.concat([df["head_p"].rename(i)   for i, df in runs.items()], axis=1)
        return (deprel.nunique(axis=1) == 1) & (head.nunique(axis=1) == 1)

    ud_mask  = _consistent_mask(ud_runs)
    str_mask = _consistent_mask(str_runs)
    consistent_mask = ud_mask & str_mask

    n_total = len(consistent_mask)
    print(f"[filter] str-new consistent: {str_mask.sum()}/{n_total} ({100*str_mask.mean():.1f}%)")
    print(f"[filter] ud-new  consistent: {ud_mask.sum()}/{n_total} ({100*ud_mask.mean():.1f}%)")
    print(f"[filter] both    consistent: {consistent_mask.sum()}/{n_total} ({100*consistent_mask.mean():.1f}%)")

    result = {}
    for corpus, runs in data.items():
        if not runs:
            continue
        first_df = next(iter(runs.values()))
        result[corpus] = first_df[consistent_mask]

    return result
