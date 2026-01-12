import pandas as pd
import numpy as np

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
    
    if "deprel_p" in merged.columns:
        merged = merged.drop(columns=["deprel_p"])
    merged = merged.rename(columns={"deprel_g": "deprel"})
    
    merged = merged.set_index(["sent_id", "id"]).sort_index()
    
    cols = ["text", "form", "deprel", "upos_g", "upos_p", "feats_g", "feats_p"]
    
    return merged[cols]


def feats_to_dict(s: str) -> dict:
    if pd.isna(s) or not s or s == "_":
        return {}
    d = {}
    for kv in str(s).split("|"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            if k:
                d[k] = v
    return d

def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["eq_upos"]  = (out["upos_g"] == out["upos_p"])
    out["eq_feats"] = (out["feats_g"].fillna("") == out["feats_p"].fillna(""))
    out["eq_all"]   = out["eq_upos"] & out["eq_feats"]
    return out

def split_summary(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    upos = (df["upos_g"] == df["upos_p"]).mean().round(4) if n else np.nan
    feats = (df["feats_g"].fillna("") == df["feats_p"].fillna("")).mean().round(4) if n else np.nan
    both = ((df["upos_g"] == df["upos_p"]) & (df["feats_g"].fillna("") == df["feats_p"].fillna(""))).mean().round(4) if n else np.nan
    return pd.Series({"rows": n, "UPOS_acc": upos, "FEATS_acc": feats, "AllTags_acc": both})

def find_most_frequent_errors(str, ud, feature="upos"):
    merged = str.merge(ud, on=["sent_id", "id"], suffixes=("_str", "_ud"))

    if feature == "upos":
        mismatch_condition = (merged["upos_g_str"] == merged["upos_p_str"]) & (merged["upos_g_ud"] != merged["upos_p_ud"])
    elif feature == "feats":
        mismatch_condition = (merged["feats_g_str"] == merged["feats_p_str"]) & (merged["feats_g_ud"] != merged["feats_p_ud"])
    else:
        raise ValueError("Feature must be 'upos' or 'feats'.")
    
    mismatches = merged[mismatch_condition]
    freq_errors = mismatches.groupby([f"{feature}_g_str", f"{feature}_g_ud", f"{feature}_p_ud"]).size().reset_index(name="count")
    freq_errors = freq_errors.sort_values(by="count", ascending=False)
    return freq_errors

def extract_error_group(str_df, ud_df, freq_errors, feature="upos", group_index=0):
    row = freq_errors.iloc[group_index]
    correct = row[f"{feature}_g_str"]
    gold_ud = row[f"{feature}_g_ud"]
    pred_ud = row[f"{feature}_p_ud"]

    merged = str_df.merge(ud_df, on=["sent_id", "id"], suffixes=("_str", "_ud"))

    if feature == "upos":
        base_mask = (merged["upos_g_str"] == merged["upos_p_str"]) & \
                    (merged["upos_g_ud"] != merged["upos_p_ud"])

        group_mask = (merged["upos_g_str"] == correct) & (
            ((merged["upos_g_ud"] == gold_ud) & (merged["upos_p_ud"] == pred_ud)) |
            ((merged["upos_g_ud"] == pred_ud) & (merged["upos_p_ud"] == gold_ud))
        )

    else:
        fg_str = merged["feats_g_str"].fillna("").replace("_", "")
        fp_str = merged["feats_p_str"].fillna("").replace("_", "")
        fg_ud  = merged["feats_g_ud"].fillna("").replace("_", "")
        fp_ud  = merged["feats_p_ud"].fillna("").replace("_", "")

        base_mask = (fg_str == fp_str) & (fg_ud != fp_ud)

        group_mask = (fg_str == correct) & (
            ((fg_ud == gold_ud) & (fp_ud == pred_ud)) |
            ((fg_ud == pred_ud) & (fp_ud == gold_ud))
        )

    examples = merged.loc[base_mask & group_mask].copy()
    examples = examples.sort_values(["sent_id", "id"]).reset_index(drop=True)

    return examples

def save_error_examples_to_csv(df, path):
    drop_cols = [
        "upos_p_str",
        "feats_g_str",
        "feats_p_str",
        "text_ud",
        "form_ud",
        "feats_g_ud",
        "feats_p_ud",
    ]

    df_out = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df_out = df_out.rename(columns={"text_str" : "text", "form_str" : "form"})
    df_out.to_csv(path, encoding="utf-8")
    return df_out

def build_upos_error_matrix(freq_dict, min_total_count=1, top_n=None):
    frames = []
    for split_name, df in freq_dict.items():
        tmp = df.copy()
        tmp["split"] = split_name
        frames.append(tmp)

    all_err = pd.concat(frames, ignore_index=True)

    all_err = (
        all_err
        .groupby(["upos_g_str", "upos_g_ud", "upos_p_ud", "split"], as_index=False)["count"]
        .sum()
    )

    mat = all_err.pivot_table(
        index=["upos_g_str", "upos_g_ud", "upos_p_ud"],
        columns="split",
        values="count",
        fill_value=0,
        aggfunc="sum",
    )

    mat["total"] = mat.sum(axis=1)

    if min_total_count is not None:
        mat = mat[mat["total"] >= min_total_count]

    mat = mat.sort_values("total", ascending=False)

    if top_n is not None:
        mat = mat.head(top_n)

    cols = [c for c in ["full", "old", "new"] if c in mat.columns] + \
           [c for c in mat.columns if c not in ("full", "old", "new")]
    mat = mat[cols]

    return mat

def get_error_sentences(str_df, ud_df,
                        upos_g_str, upos_g_ud, upos_p_ud,
                        with_sentences=False):
    if "sent_id" not in str_df.columns or "id" not in str_df.columns:
        str_df = str_df.reset_index()
    if "sent_id" not in ud_df.columns or "id" not in ud_df.columns:
        ud_df = ud_df.reset_index()

    merged = str_df.merge(ud_df, on=["sent_id", "id"], suffixes=("_str", "_ud"))

    base_mask = (merged["upos_g_str"] == merged["upos_p_str"]) & \
                (merged["upos_g_ud"] != merged["upos_p_ud"])

    type_mask = (
        (merged["upos_g_str"] == upos_g_str) &
        (merged["upos_g_ud"] == upos_g_ud) &
        (merged["upos_p_ud"] == upos_p_ud)
    )

    errors = merged.loc[base_mask & type_mask].copy()
    errors = errors.reset_index(drop=True)
    errors = errors.sort_values(["sent_id", "id"])

    return errors

def browse_upos_errors(df):
    n = len(df)
    print(f"Всего примеров: {n}")

    for i, row in df.iterrows():
        print("=" * 80)
        print(f"Пример #{i}  (sent_id={row.get('sent_id')}, id={row.get('id')})")
        print("-" * 80)

        sent_text = (
            row.get("text_ud")
            or row.get("text_str")
            or row.get("text")
        )
        if sent_text is not None:
            print("Предложение:")
            print(sent_text)
        else:
            print("Предложение (собрано по form_str/form):")
            form_col = "form_str" if "form_str" in df.columns else "form"
            sent_tokens = df[df["sent_id"] == row["sent_id"]][form_col].tolist()
            print(" ".join(map(str, sent_tokens)))

        print("-" * 80)

        if "form_str" in df.columns:
            form = row["form_str"]
        elif "form_ud" in df.columns:
            form = row["form_ud"]
        else:
            form = row.get("form")

        print(f"Токен: {form!r}")

        print("UPOS:")
        print(f"  SynTagRus gold (upos_g_str): {row.get('upos_g_str')}")
        print(f"  UD gold      (upos_g_ud):  {row.get('upos_g_ud')}")
        print(f"  UD pred      (upos_p_ud):  {row.get('upos_p_ud')}")
        
        print("DEPREL:")
        print(f"  SynTagRus: {row.get('deprel_str')}")
        print(f"  UD:        {row.get('deprel_ud')}")

        if "feats_g_str" in df.columns or "feats_g_ud" in df.columns or "feats_p_ud" in df.columns:
            print("FEATS:")
            if "feats_g_str" in df.columns:
                print(f"  SynTagRus gold (feats_g_str): {row.get('feats_g_str')}")
            if "feats_g_ud" in df.columns:
                print(f"  UD gold       (feats_g_ud):  {row.get('feats_g_ud')}")
            if "feats_p_ud" in df.columns:
                print(f"  UD pred       (feats_p_ud):  {row.get('feats_p_ud')}")

        print("=" * 80)
        inp = input("Нажми Enter для следующего примера (или q + Enter, чтобы выйти): ")
        if inp.strip().lower() == "q":
            break

