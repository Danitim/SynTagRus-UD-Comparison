import pandas as pd
import numpy as np
from pathlib import Path

def find_pairs_to_swap(str_df, ud_df):
    str_df = str_df.copy()
    ud_df = ud_df.copy()

    merged = str_df[["sent_id", "id", "head"]].merge(
        ud_df[["sent_id", "id", "head"]],
        on=["sent_id", "id"],
        suffixes=("_str", "_ud")
    )
    
    merged_head = merged.rename(columns={
        "id": "id_head",
        "head_str": "head_str_at_head",
        "head_ud": "head_ud_at_head"
    })

    pairs_df = merged.merge(
        merged_head[["sent_id", "id_head", "head_ud_at_head"]],
        left_on=["sent_id", "head_str"],
        right_on=["sent_id", "id_head"],
        how="inner"
    )

    pairs_df = pairs_df[
        (pairs_df["head_ud_at_head"].notna()) &
        (pairs_df["head_ud_at_head"] == pairs_df["id"])
    ].copy()

    pairs_df["id1"] = pairs_df["head_str"].astype(int)
    pairs_df["id2"] = pairs_df["id"].astype(int)

    pairs_df["id_min"] = pairs_df[["id1", "id2"]].min(axis=1)
    pairs_df["id_max"] = pairs_df[["id1", "id2"]].max(axis=1)

    result = pairs_df[["sent_id", "id_min", "id_max"]].drop_duplicates()

    return [(row["sent_id"], int(row["id_min"]), int(row["id_max"]))
            for _, row in result.iterrows()]


def swap_rows_in_df(df, sent_id, id1, id2):
    df = df.copy()

    mask1 = (df["sent_id"] == sent_id) & (df["id"] == id1)
    mask2 = (df["sent_id"] == sent_id) & (df["id"] == id2)

    if not mask1.any() or not mask2.any():
        return df

    idx1 = df[mask1].index[0]
    idx2 = df[mask2].index[0]

    cols_to_swap = [c for c in df.columns if c not in ["sent_id", "id", "head"]]

    temp = df.loc[idx1, cols_to_swap].copy()
    df.loc[idx1, cols_to_swap] = df.loc[idx2, cols_to_swap].values
    df.loc[idx2, cols_to_swap] = temp.values

    sent_mask = df["sent_id"] == sent_id

    head1 = df.loc[idx1, "head"]
    head2 = df.loc[idx2, "head"]

    if head1 == id2:
        boss_id, dep_id = id2, id1
        boss_idx, dep_idx = idx2, idx1
        boss_head = head2
    elif head2 == id1:
        boss_id, dep_id = id1, id2
        boss_idx, dep_idx = idx1, idx2
        boss_head = head1
    else:
        return df

    df.loc[boss_idx, "head"] = dep_id
    df.loc[dep_idx, "head"] = boss_head

    for idx in df[sent_mask].index:
        if idx not in [idx1, idx2]:
            if df.loc[idx, "head"] == boss_id:
                df.loc[idx, "head"] = dep_id
            elif df.loc[idx, "head"] == dep_id:
                df.loc[idx, "head"] = boss_id
                

    return df


def align_markup(datasets, input_path="csvs", output_path="aligned", verbose=True):
    input_dir = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)

    for split, items in datasets.items():
        if verbose:
            print(f"Aligning markup for dataset: {items} (split: {split})")
        str_gold_path = input_dir / f"{items[0]}.test.csv"
        str_pred_path = input_dir / f"{items[0]}.test.pred.csv"
        ud_gold_path = input_dir / f"{items[1]}.test.csv"
        ud_pred_path = input_dir / f"{items[1]}.test.pred.csv"

        str_gold_df = pd.read_csv(str_gold_path, dtype={"sent_id": str})
        str_pred_df = pd.read_csv(str_pred_path, dtype={"sent_id": str})
        ud_gold_df = pd.read_csv(ud_gold_path, dtype={"sent_id": str})
        ud_pred_df = pd.read_csv(ud_pred_path, dtype={"sent_id": str})

        pairs = find_pairs_to_swap(str_gold_df, ud_gold_df)
        if verbose:
            print(f"  Found {len(pairs)} pairs to swap.")

        for sent_id, id1, id2 in pairs:
            str_gold_df = swap_rows_in_df(str_gold_df, sent_id, id1, id2)
            str_pred_df = swap_rows_in_df(str_pred_df, sent_id, id1, id2)

        str_gold_df.to_csv(output_dir / f"{items[0]}.test.csv", index=False, encoding="utf-8")
        str_pred_df.to_csv(output_dir / f"{items[0]}.test.pred.csv", index=False, encoding="utf-8")
        ud_gold_df.to_csv(output_dir / f"{items[1]}.test.csv", index=False, encoding="utf-8")
        ud_pred_df.to_csv(output_dir / f"{items[1]}.test.pred.csv", index=False, encoding="utf-8")
        
        if verbose:
            print(f"  Saved aligned files to {output_dir}")
