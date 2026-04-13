import pandas as pd
import numpy as np


def build_error_matrix(data: dict, sort_by: str = "count", min_count: int = 10, min_pct: float = 1.0) -> pd.DataFrame:
    """
    Build an error matrix for the CV dataset (str-new vs ud-new only).

    Input: data from filter_consistent → {"str-new": df, "ud-new": df}

    Finds tokens where STR prediction is correct but UD prediction is wrong,
    grouped by (deprel_g_str, deprel_g_ud, deprel_p_ud).

    Columns: count, pct (% of this (str_gold, ud_gold) pair that has this error).
    """
    str_df = data.get("str-new")
    ud_df  = data.get("ud-new")

    if str_df is None or ud_df is None:
        return pd.DataFrame()

    str_df = str_df.reset_index()
    ud_df  = ud_df.reset_index()

    merged = str_df.merge(ud_df, on=["sent_id", "id"], suffixes=("_str", "_ud"))

    str_correct   = merged["deprel_g_str"] == merged["deprel_p_str"]
    ud_incorrect  = merged["deprel_g_ud"]  != merged["deprel_p_ud"]

    errors = merged[str_correct & ud_incorrect].copy()

    error_counts = (
        errors
        .groupby(["deprel_g_str", "deprel_g_ud", "deprel_p_ud"])
        .size()
        .reset_index(name="count")
    )

    # Denominator: total tokens per (deprel_g_str, deprel_g_ud) pair
    pair_totals = (
        merged
        .groupby(["deprel_g_str", "deprel_g_ud"])
        .size()
        .to_dict()
    )

    error_counts["pct"] = error_counts.apply(
        lambda row: round(
            row["count"] / pair_totals.get((row["deprel_g_str"], row["deprel_g_ud"]), 1) * 100,
            2,
        ),
        axis=1,
    )

    error_matrix = error_counts.set_index(["deprel_g_str", "deprel_g_ud", "deprel_p_ud"])
    error_matrix["count"] = error_matrix["count"].astype("Int64")

    # Filter by minimum count and percentage
    mask = (error_matrix["count"] > min_count) & (error_matrix["pct"] > min_pct)
    error_matrix = error_matrix[mask]

    error_matrix = error_matrix.sort_values(sort_by, ascending=False)

    return error_matrix
