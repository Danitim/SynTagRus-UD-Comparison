import pandas as pd
import numpy as np


def build_error_matrix(data, sort_by="total"):
    splits_mapping = {
        "full": ("str", "ud"),
        "new": ("str-new", "ud-new"),
        "old": ("str-old", "ud-old")
    }

    all_errors = []

    for split_name, (str_key, ud_key) in splits_mapping.items():
        if str_key not in data or ud_key not in data:
            continue

        str_df = data[str_key].reset_index()
        ud_df = data[ud_key].reset_index()

        merged = str_df.merge(ud_df, on=["sent_id", "id"], suffixes=("_str", "_ud"))

        str_correct = merged["deprel_g_str"] == merged["deprel_p_str"]
        ud_incorrect = merged["deprel_g_ud"] != merged["deprel_p_ud"]

        errors = merged[str_correct & ud_incorrect].copy()

        error_counts = (
            errors.groupby(["deprel_g_str", "deprel_g_ud", "deprel_p_ud"])
            .size()
            .reset_index(name="count")
        )
        error_counts["split"] = split_name

        all_errors.append(error_counts)

    if not all_errors:
        return pd.DataFrame()

    all_errors_df = pd.concat(all_errors, ignore_index=True)

    error_matrix = all_errors_df.pivot_table(
        index=["deprel_g_str", "deprel_g_ud", "deprel_p_ud"],
        columns="split",
        values="count",
        fill_value=0,
        aggfunc="sum"
    )

    error_matrix["total"] = error_matrix.sum(axis=1)

    pair_totals = {}
    for split_name, (str_key, ud_key) in splits_mapping.items():
        if str_key not in data or ud_key not in data:
            continue

        str_df = data[str_key].reset_index()
        ud_df = data[ud_key].reset_index()

        merged = str_df.merge(ud_df, on=["sent_id", "id"], suffixes=("_str", "_ud"))

        totals = (
            merged.groupby(["deprel_g_str", "deprel_g_ud"])
            .size()
            .to_dict()
        )
        pair_totals[split_name] = totals

    for split_name in splits_mapping.keys():
        if split_name in error_matrix.columns:
            pct_col = f"{split_name}_pct"
            error_matrix[pct_col] = error_matrix.apply(
                lambda row: round((row[split_name] / pair_totals[split_name].get(
                    (row.name[0], row.name[1]), 1
                )) * 100, 2),
                axis=1
            )

    for split_name in splits_mapping.keys():
        if split_name not in error_matrix.columns:
            continue
        totals = pair_totals.get(split_name, {})
        pair_keys = error_matrix.index.droplevel(2)
        pair_missing = pd.Series(
            [key not in totals for key in pair_keys],
            index=error_matrix.index,
        )
        error_matrix.loc[pair_missing, split_name] = np.nan
        pct_col = f"{split_name}_pct"
        if pct_col in error_matrix.columns:
            error_matrix.loc[pair_missing, pct_col] = np.nan

    for col in ["full", "old", "new", "total"]:
        if col in error_matrix.columns:
            error_matrix[col] = error_matrix[col].astype("Int64")

    cols_order = []
    for split_name in ["full", "old", "new"]:
        if split_name in error_matrix.columns:
            cols_order.append(split_name)
            pct_col = f"{split_name}_pct"
            if pct_col in error_matrix.columns:
                cols_order.append(pct_col)

    if "total" in error_matrix.columns:
        cols_order.append("total")

    other_cols = [c for c in error_matrix.columns if c not in cols_order]
    error_matrix = error_matrix[cols_order + other_cols]

    mask_count = (error_matrix["full"] > 10) | (error_matrix["old"] > 10) | (error_matrix["new"] > 10)
    mask_percent = (error_matrix["full_pct"] > 1.0) | (error_matrix["old_pct"] > 1.0) | (error_matrix["new_pct"] > 1.0)
    error_matrix = error_matrix[mask_count & mask_percent]
    
    error_matrix = error_matrix.sort_values(sort_by, ascending=False)

    return error_matrix
