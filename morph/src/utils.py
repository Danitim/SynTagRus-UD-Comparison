import pandas as pd
import numpy as np
from pathlib import Path

def load_pair(gold_csv, pred_csv):
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
    
    if "head_p" in merged.columns:
        merged = merged.drop(columns=["head_p"])
    merged = merged.rename(columns={"head_g": "head"})
    
    if "deprel_p" in merged.columns:
        merged = merged.drop(columns=["deprel_p"])
    merged = merged.rename(columns={"deprel_g": "deprel"})
    
    if "ellipsis_p" in merged.columns:
        merged = merged.drop(columns=["ellipsis_p"])
    if "ellipsis_g" in merged.columns:
        merged = merged.rename(columns={"ellipsis_g": "ellipsis"})

    if "lemma_p" in merged.columns:
        merged = merged.drop(columns=["lemma_p"])
    if "lemma_g" in merged.columns:
        merged = merged.rename(columns={"lemma_g": "lemma"})

    merged = merged.set_index(["sent_id", "id"]).sort_index()

    cols = [
        "text",
        "form",
        "lemma",
        "upos_p",
        "upos_g",
        "feats_p",
        "feats_g",
        "head",
        "deprel",
        "ellipsis"
    ]
    
    return merged[cols]


def build_data_from_suffix(
    suffix: str,
    *,
    csv_dir: str = "csvs",
):
    from src.convert import convert_conllu_list_to_csv

    manifest = [
        ("ud",      "gold", "../datasets/ud/test.conllu"),
        ("ud",      "pred", f"../out/ud-{suffix}/test.pred.conllu"),
        ("ud-new",  "gold", "../datasets/ud-new/test.conllu"),
        ("ud-new",  "pred", f"../out/ud-new-{suffix}/test.pred.conllu"),
        ("ud-old",  "gold", "../datasets/ud-old/test.conllu"),
        ("ud-old",  "pred", f"../out/ud-old-{suffix}/test.pred.conllu"),
        ("str",     "gold", "../datasets/str/test.conllu"),
        ("str",     "pred", f"../out/str-{suffix}/test.pred.conllu"),
        ("str-new", "gold", "../datasets/str-new/test.conllu"),
        ("str-new", "pred", f"../out/str-new-{suffix}/test.pred.conllu"),
        ("str-old", "gold", "../datasets/str-old/test.conllu"),
        ("str-old", "pred", f"../out/str-old-{suffix}/test.pred.conllu"),
    ]

    inputs = [p for _, _, p in manifest]
    csvs = convert_conllu_list_to_csv(inputs, csv_dir)

    csv_index = {}
    for (split, role, _), csv in zip(manifest, csvs):
        csv_index.setdefault(split, {})[role] = Path(csv)

    data = {split: load_pair(paths["gold"], paths["pred"]) for split, paths in csv_index.items()}
    return data


def _extract_feat_value(feats, feat_name: str) -> str:
    if pd.isna(feats) or feats in ("", "_"):
        return "∅"
    for pair in str(feats).split("|"):
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        if key == feat_name:
            return value
    return "∅"


def save_str_ud_morph_mismatches(
    data: dict,
    split: str,
    tag: str,
    str_tag: str = None,
    ud_g_tag: str = None,
    ud_p_tag: str = None,
    out_path: str = None,
    index: int = None,
) -> pd.DataFrame:
    """
    Save all mismatches for the selected tag where STR is correct and UD is wrong.
    `tag` can be "upos" or a single UD feature name (for example "Case", "Gender").
    """
    if split not in {"full", "old", "new"}:
        raise ValueError("split must be one of: 'full', 'old', 'new'.")
    if tag == "feats":
        raise ValueError("For FEATS, pass a concrete feature name (for example 'Case').")

    str_df = data["str"] if split == "full" else data[f"str-{split}"]
    ud_df = data["ud"] if split == "full" else data[f"ud-{split}"]

    if "sent_id" not in str_df.columns or "id" not in str_df.columns:
        str_df = str_df.reset_index()
    if "sent_id" not in ud_df.columns or "id" not in ud_df.columns:
        ud_df = ud_df.reset_index()

    merged = str_df.merge(ud_df, on=["sent_id", "id"], suffixes=("_str", "_ud"))

    mism = merged[
        (merged[f"{tag}_g_str"] == merged[f"{tag}_p_str"])
        & (merged[f"{tag}_g_ud"] != merged[f"{tag}_p_ud"])
    ].copy()
    
    mism = mism[mism[f"{tag}_g_str"] == str_tag]
    mism = mism[mism[f"{tag}_g_ud"] == ud_g_tag]
    mism = mism[mism[f"{tag}_p_ud"] == ud_p_tag]
    
    # print(mism.columns.tolist())

    out = mism[
        [
            "sent_id",
            "text_str",
            "id",
            "form_ud",
            "upos_g_str",
            "feats_g_str",
            "deprel_str",
            "upos_g_ud",
            "feats_g_ud",
            "upos_p_ud",
            "feats_p_ud",
            "deprel_ud",
        ]
    ].rename(
        columns={
            "text_str": "text",
            "form_ud": "form",
            "upos_g_str": "upos_str",
            "feats_g_str": "feats_str",
        }
    )

    lines = []
    for _, row in out.iterrows():
        lines.append(
            "sent_id: {sent_id}\n"
            "text: {text}\n"
            "id: {id}\n"
            "form: {form}\n"
            "upos_str: {upos_str}\n"
            "feats_str: {feats_str}\n"
            "deprel_str: {deprel_str}\n"
            "----------\n"
            "upos_g_ud: {upos_g_ud}\n"
            "feats_g_ud: {feats_g_ud}\n"
            "VS\n"
            "upos_p_ud: {upos_p_ud}\n"
            "feats_p_ud: {feats_p_ud}\n"
            "----------\n"
            "deprel: {deprel_ud}\n"
            "================================================="
            "\n".format(**row)
        )

    if out_path is None:
        if index is not None:
            out_path = f"data_{split}_{index}_{tag}.txt"
        else:
            out_path = f"data_{split}_{tag}_{str_tag}_{ud_g_tag}_{ud_p_tag}.txt"

    Path(out_path).write_text("".join(lines), encoding="utf-8")
    return out


def save_sig_cases(
    token_df: pd.DataFrame,
    sig: tuple,
    out_path: str = None,
) -> pd.DataFrame:
    """
    Save all tokens from results["tokens"] matching a given error signature to a text file.

    Parameters
    ----------
    token_df : results["tokens"] — filtered, annotated token dataframe with a "sig" column.
    sig      : exact signature tuple, e.g. (("Case", "ИМ", "Nom", "∅"),).
               Each element is (c_ud, str_gold, ud_gold, ud_pred).
               Must match the tuples stored in the "sig" column.
    out_path : output file path. Defaults to "sig_<categories>.txt".
    """
    subset = token_df[token_df["sig"] == sig].copy()

    if out_path is None:
        parts = "+".join(f"{c}_{sg}_{ug}_{up}" for c, sg, ug, up in sig)
        out_path = f"sig_{parts}.txt"

    lines = []
    for _, row in subset.iterrows():
        lines.append(
            "sent_id: {sent_id}\n"
            "text: {text}\n"
            "id: {id}\n"
            "form: {form}\n"
            "--- STR gold ---\n"
            "upos: {upos_g_str}\n"
            "feats: {feats_g_str}\n"
            "--- UD gold ---\n"
            "upos: {upos_g_ud}\n"
            "feats: {feats_g_ud}\n"
            "--- UD pred ---\n"
            "upos: {upos_p_ud}\n"
            "feats: {feats_p_ud}\n"
            "=================================================\n".format(
                sent_id=row["sent_id"],
                text=row.get("text_str", row.get("text_ud", "")),
                id=row["id"],
                form=row.get("form_str", row.get("form_ud", "")),
                upos_g_str=row["upos_g_str"],
                feats_g_str=row["feats_g_str"],
                upos_g_ud=row["upos_g_ud"],
                feats_g_ud=row["feats_g_ud"],
                upos_p_ud=row["upos_p_ud"],
                feats_p_ud=row["feats_p_ud"],
            )
        )

    Path(out_path).write_text("".join(lines), encoding="utf-8")
    return subset
