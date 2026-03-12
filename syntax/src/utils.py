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
    df: pd.DataFrame,
    split: str,
    deprel_str: str,
    deprel_ud_gold: str,
    deprel_ud_predicted: str,
    out_path: str = None,
    index: int = None
) -> pd.DataFrame:
    
    str_df = df[f"str-{split}"] if split != "full" else df["str"]
    ud_df = df[f"ud-{split}"] if split != "full" else df["ud"]
    
    if "sent_id" not in str_df.columns or "id" not in str_df.columns:
        str_df = str_df.reset_index()
    if "sent_id" not in ud_df.columns or "id" not in ud_df.columns:
        ud_df = ud_df.reset_index()

    merged = str_df.merge(ud_df, on=["sent_id", "id"], suffixes=("_str", "_ud"))

    mism = merged[merged["deprel_g_ud"] != merged["deprel_p_ud"]].copy()

    if deprel_str == "any":
        pass  # no STR filter; sort later by duplets
    elif deprel_str == "anyBut":
        mism = mism[mism["deprel_g_str"] != mism["deprel_p_str"]]
    elif deprel_str is not None:
        mism = mism[mism["deprel_g_str"] == deprel_str]

    if deprel_ud_gold is not None:
        mism = mism[mism["deprel_g_ud"] == deprel_ud_gold]
    if deprel_ud_predicted is not None:
        mism = mism[mism["deprel_p_ud"] == deprel_ud_predicted]

    head_lookup = ud_df[["sent_id", "id", "form", "upos", "feats"]].rename(
        columns={
            "id": "head_id",
            "form": "head_form",
            "upos": "head_upos",
            "feats": "head_feats",
        }
    )
    mism = mism.merge(
        head_lookup,
        left_on=["sent_id", "head_g_ud"],
        right_on=["sent_id", "head_id"],
        how="left",
    )

    if deprel_str == "any":
        mism = mism.sort_values(["deprel_g_ud", "deprel_p_ud"])
        out = mism[
            [
                "sent_id",
                "text_str",
                "id",
                "form_ud",
                "upos_ud",
                "feats_ud",
                "deprel_g_ud",
                "deprel_p_ud",
                "head_form",
                "head_upos",
                "head_feats",
            ]
        ].rename(
            columns={
                "text_str": "text",
                "form_ud": "form",
                "upos_ud": "upos",
                "feats_ud": "feats",
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
                "upos: {upos}\n"
                "feats: {feats}\n"
                "deprel_ud_gold: {deprel_ud_gold}\n"
                "deprel_ud_predicted: {deprel_ud_predicted}\n"
                "head_form: {head_form}\n"
                "head_upos: {head_upos}\n"
                "head_feats: {head_feats}\n"
                "----\n".format(**row)
            )
    elif deprel_str == "anyBut":
        mism = mism.sort_values(["deprel_p_str", "deprel_g_ud", "deprel_p_ud"])
        out = mism[
            [
                "sent_id",
                "text_str",
                "id",
                "form_ud",
                "upos_ud",
                "feats_ud",
                "upos_str",
                "feats_str",
                "deprel_p_str",
                "deprel_g_ud",
                "deprel_p_ud",
                "head_form",
                "head_upos",
                "head_feats",
            ]
        ].rename(
            columns={
                "text_str": "text",
                "form_ud": "form",
                "upos_ud": "upos",
                "feats_ud": "feats",
                "upos_str": "upos_str",
                "feats_str": "feats_str",
                "deprel_p_str": "deprel_str_predicted",
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
                "upos: {upos}\n"
                "feats: {feats}\n"
                "upos_str: {upos_str}\n"
                "feats_str: {feats_str}\n"
                "deprel_str_predicted: {deprel_str_predicted}\n"
                "deprel_ud_gold: {deprel_ud_gold}\n"
                "deprel_ud_predicted: {deprel_ud_predicted}\n"
                "head_form: {head_form}\n"
                "head_upos: {head_upos}\n"
                "head_feats: {head_feats}\n"
                "----\n".format(**row)
            )
    else:
        out = mism[
            [
                "sent_id",
                "text_str",
                "id",
                "form_ud",
                "upos_ud",
                "feats_ud",
                "upos_str",
                "feats_str",
                "deprel_g_str",
                "deprel_g_ud",
                "deprel_p_ud",
                "head_form",
                "head_upos",
                "head_feats",
            ]
        ].rename(
            columns={
                "text_str": "text",
                "form_ud": "form",
                "upos_ud": "upos",
                "feats_ud": "feats",
                "upos_str": "upos_str",
                "feats_str": "feats_str",
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
                "upos: {upos}\n"
                "feats: {feats}\n"
                "upos_str: {upos_str}\n"
                "feats_str: {feats_str}\n"
                "deprel_str: {deprel_str}\n"
                "deprel_ud_gold: {deprel_ud_gold}\n"
                "deprel_ud_predicted: {deprel_ud_predicted}\n"
                "head_form: {head_form}\n"
                "head_upos: {head_upos}\n"
                "head_feats: {head_feats}\n"
                "----\n".format(**row)
            )

    if out_path is None:
        if deprel_str in ("any", "anyBut"):
            str_name = deprel_str
        else:
            str_name = "".join(c for c in deprel_str if c.isalnum())
        ud_gold_name = "".join(c for c in deprel_ud_gold if c.isalnum()) if deprel_ud_gold else "any"
        ud_pred_name = "".join(c for c in deprel_ud_predicted if c.isalnum()) if deprel_ud_predicted else "any"
        if index is not None:
            out_path = f"data_{split}_{index}_{str_name}_{ud_gold_name}_{ud_pred_name}.txt"
        else:
            out_path = f"data_{split}_{str_name}_{ud_gold_name}_{ud_pred_name}.txt"
    Path(out_path).write_text("".join(lines), encoding="utf-8")
    return out


def save_noisy_triplet(
    df: pd.DataFrame,
    split: str,
    deprel_str_predicted: str,
    deprel_ud_gold: str,
    deprel_ud_predicted: str,
    out_path: str = None,
    index: int = None,
) -> pd.DataFrame:
    """
    Save examples for a noisy triplet (str_p, ud_g, ud_p), i.e. rows where
      - deprel_p_str == deprel_str_predicted  (the wrong STR prediction)
      - deprel_g_str != deprel_p_str          (STR prediction is indeed wrong)
      - deprel_g_ud  == deprel_ud_gold
      - deprel_p_ud  == deprel_ud_predicted

    Rows are sorted by deprel_g_str so examples from each true gold STR label
    appear together.
    """
    str_df = df["str"]      if split == "full" else df[f"str-{split}"]
    ud_df  = df["ud"]       if split == "full" else df[f"ud-{split}"]

    if "sent_id" not in str_df.columns or "id" not in str_df.columns:
        str_df = str_df.reset_index()
    if "sent_id" not in ud_df.columns or "id" not in ud_df.columns:
        ud_df = ud_df.reset_index()

    merged = str_df.merge(ud_df, on=["sent_id", "id"], suffixes=("_str", "_ud"))

    mism = merged[
        (merged["deprel_p_str"] == deprel_str_predicted)
        & (merged["deprel_g_str"] != merged["deprel_p_str"])
        & (merged["deprel_g_ud"]  == deprel_ud_gold)
        & (merged["deprel_p_ud"]  == deprel_ud_predicted)
    ].copy()

    head_lookup = ud_df[["sent_id", "id", "form", "upos", "feats"]].rename(
        columns={
            "id":    "head_id",
            "form":  "head_form",
            "upos":  "head_upos",
            "feats": "head_feats",
        }
    )
    mism = mism.merge(
        head_lookup,
        left_on=["sent_id", "head_g_ud"],
        right_on=["sent_id", "head_id"],
        how="left",
    )

    mism = mism.sort_values("deprel_g_str")

    out = mism[
        [
            "sent_id",
            "text_str",
            "id",
            "form_ud",
            "upos_ud",
            "feats_ud",
            "upos_str",
            "feats_str",
            "deprel_g_str",
            "deprel_p_str",
            "deprel_g_ud",
            "deprel_p_ud",
            "head_form",
            "head_upos",
            "head_feats",
        ]
    ].rename(
        columns={
            "text_str":   "text",
            "form_ud":    "form",
            "upos_ud":    "upos",
            "feats_ud":   "feats",
            "upos_str":   "upos_str",
            "feats_str":  "feats_str",
            "deprel_g_str": "deprel_str",
            "deprel_p_str": "deprel_str_predicted",
            "deprel_g_ud":  "deprel_ud_gold",
            "deprel_p_ud":  "deprel_ud_predicted",
        }
    )

    lines = []
    for _, row in out.iterrows():
        lines.append(
            "sent_id: {sent_id}\n"
            "text: {text}\n"
            "id: {id}\n"
            "form: {form}\n"
            "upos: {upos}\n"
            "feats: {feats}\n"
            "upos_str: {upos_str}\n"
            "feats_str: {feats_str}\n"
            "deprel_str: {deprel_str}\n"
            "deprel_str_predicted: {deprel_str_predicted}\n"
            "deprel_ud_gold: {deprel_ud_gold}\n"
            "deprel_ud_predicted: {deprel_ud_predicted}\n"
            "head_form: {head_form}\n"
            "head_upos: {head_upos}\n"
            "head_feats: {head_feats}\n"
            "----\n".format(**row)
        )

    if out_path is None:
        sp_name  = "".join(c for c in deprel_str_predicted if c.isalnum())
        udg_name = "".join(c for c in deprel_ud_gold      if c.isalnum())
        udp_name = "".join(c for c in deprel_ud_predicted if c.isalnum())
        if index is not None:
            out_path = f"data_{split}_{index}_noisy_{sp_name}_{udg_name}_{udp_name}.txt"
        else:
            out_path = f"data_{split}_noisy_{sp_name}_{udg_name}_{udp_name}.txt"

    Path(out_path).write_text("".join(lines), encoding="utf-8")
    return out


def save_inconsistency(
    data: dict,
    deprel_str: str,
    ud_labels: tuple,
    out_path: str = None,
):
    """Save examples from old and new UD gold where a fixed STR deprel
    maps to any of the given UD labels. Works for any deprel pair."""

    def _merge_split(str_key, ud_key, split_name):
        str_df = data[str_key].reset_index()
        ud_df = data[ud_key].reset_index()
        merged = str_df.merge(ud_df, on=["sent_id", "id"], suffixes=("_str", "_ud"))
        subset = merged[merged["deprel_g_str"] == deprel_str].copy()
        subset = subset[subset["deprel_g_ud"].isin(ud_labels)]
        subset["split"] = split_name

        head_lookup = ud_df[["sent_id", "id", "form", "upos", "feats"]].rename(
            columns={
                "id": "head_id",
                "form": "head_form",
                "upos": "head_upos",
                "feats": "head_feats",
            }
        )
        subset = subset.merge(
            head_lookup,
            left_on=["sent_id", "head_g_ud"],
            right_on=["sent_id", "head_id"],
            how="left",
        )
        return subset

    old_df = _merge_split("str-old", "ud-old", "old")
    new_df = _merge_split("str-new", "ud-new", "new")

    lines = []
    for split_name, split_df in [("OLD", old_df), ("NEW", new_df)]:
        for label in ud_labels:
            examples = split_df[split_df["deprel_g_ud"] == label].head(3)
            if len(examples) == 0:
                continue
            lines.append(f"{'=' * 70}\n")
            lines.append(f"{split_name}, UD gold = {label} ({len(examples)} шт.)\n")
            lines.append(f"{'=' * 70}\n")
            for i, row in examples.iterrows():
                
                text = str(row.get("text_str", ""))
                if len(text) > 200:
                    text = text[:200] + "..."
                lines.append(
                    f"sent_id: {row['sent_id']}\n"
                    f"text: {text}\n"
                    f"id: {row['id']}\n"
                    f"form: {row['form_ud']}\n"
                    f"upos: {row['upos_ud']}\n"
                    f"feats: {row['feats_ud']}\n"
                    f"deprel_str: {row['deprel_g_str']}\n"
                    f"deprel_ud_gold: {row['deprel_g_ud']}\n"
                    f"head_form: {row['head_form']}\n"
                    f"head_upos: {row['head_upos']}\n"
                    f"head_feats: {row['head_feats']}\n"
                    f"----\n"
                )
            lines.append("\n")

    if out_path is None:
        str_name = "".join(c for c in deprel_str if c.isalnum())
        label_names = "_".join("".join(c for c in l if c.isalnum()) for l in ud_labels)
        out_path = f"data_inconsistency_{str_name}_{label_names}.txt"
    Path(out_path).write_text("".join(lines), encoding="utf-8")
