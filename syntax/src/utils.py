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
    
    cols = [
        "text",
        "form",
        "upos",
        "feats",
        "head_g",
        "head_p",
        "deprel_g",
        "deprel_p",
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


def save_str_ud_deprel_mismatches(
    str_df: pd.DataFrame,
    ud_df: pd.DataFrame,
    out_path: str,
    *,
    deprel_str: str = None,
    deprel_ud_gold: str = None,
    deprel_ud_predicted: str = None,
) -> pd.DataFrame:
    if "sent_id" not in str_df.columns or "id" not in str_df.columns:
        str_df = str_df.reset_index()
    if "sent_id" not in ud_df.columns or "id" not in ud_df.columns:
        ud_df = ud_df.reset_index()

    merged = str_df.merge(ud_df, on=["sent_id", "id"], suffixes=("_str", "_ud"))

    mism = merged[merged["deprel_g_ud"] != merged["deprel_p_ud"]].copy()
    if deprel_str is not None:
        mism = mism[mism["deprel_g_str"] == deprel_str]
    if deprel_ud_gold is not None:
        mism = mism[mism["deprel_g_ud"] == deprel_ud_gold]
    if deprel_ud_predicted is not None:
        mism = mism[mism["deprel_p_ud"] == deprel_ud_predicted]

    out = mism[
        [
            "sent_id",
            "text_str",
            "id",
            "form_str",
            "deprel_g_str",
            "deprel_g_ud",
            "deprel_p_ud",
        ]
    ].rename(
        columns={
            "text_str": "text",
            "form_str": "form",
            "deprel_g_str": "deprel_str",
            "deprel_g_ud": "deprel_ud_gold",
            "deprel_p_ud": "deprel_ud_predicted",
        }
    )

    lines = []
    for _, row in out.iterrows():
        lines.append(
            "sent_id: {sent_id}\n"
            "text: {text}\n"
            "id: {id}\n"
            "form: {form}\n"
            "deprel_str: {deprel_str}\n"
            "deprel_ud_gold: {deprel_ud_gold}\n"
            "deprel_ud_predicted: {deprel_ud_predicted}\n"
            "----\n".format(**row)
        )
    Path(out_path).write_text("".join(lines), encoding="utf-8")
    return out
