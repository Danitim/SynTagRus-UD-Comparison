"""
Alignment of SynTagRus and UD dependency annotations.

Two approaches (see method.md):

  1. Swap-based alignment (Section 2) — primary approach.
     Detects mirrored edges and swaps str data so both corpora share
     the same arc orientation, enabling correct token-level comparison.

  2. Edge-based comparison (Section 4) — auxiliary approach.
     Read-only. Classifies edges into matched (M), str-only (D_str),
     ud-only (D_ud) and compares deprels on matched edges directly.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Section 2: Swap-based alignment (primary, modifies str data)
# ---------------------------------------------------------------------------

def find_pairs_to_swap(
    str_df: pd.DataFrame,
    ud_df:  pd.DataFrame,
) -> list[tuple]:
    """
    Find mirrored pairs (Section 2.1):
      h_str(j) = i  and  h_ud(i) = j

    Returns list of (sent_id, min(i,j), max(i,j)).
    """
    merged = str_df[["sent_id", "id", "head"]].merge(
        ud_df[["sent_id", "id", "head"]],
        on=["sent_id", "id"],
        suffixes=("_str", "_ud"),
    )

    merged_head = merged.rename(columns={
        "id": "id_head",
        "head_str": "head_str_at_head",
        "head_ud": "head_ud_at_head",
    })

    pairs_df = merged.merge(
        merged_head[["sent_id", "id_head", "head_ud_at_head"]],
        left_on=["sent_id", "head_str"],
        right_on=["sent_id", "id_head"],
        how="inner",
    )

    pairs_df = pairs_df[
        pairs_df["head_ud_at_head"].notna() &
        (pairs_df["head_ud_at_head"] == pairs_df["id"])
    ].copy()

    pairs_df["id_min"] = pairs_df[["head_str", "id"]].min(axis=1).astype(int)
    pairs_df["id_max"] = pairs_df[["head_str", "id"]].max(axis=1).astype(int)

    result = pairs_df[["sent_id", "id_min", "id_max"]].drop_duplicates()
    return [(row["sent_id"], row["id_min"], row["id_max"])
            for _, row in result.iterrows()]


def swap_rows_in_df(
    df:      pd.DataFrame,
    sent_id,
    id1:     int,
    id2:     int,
) -> pd.DataFrame:
    """
    Apply swap(b, d) to df (Section 2.2).

    Step 1: exchange all attributes except sent_id, id, head.
    Step 2: redirect arcs according to the formula for h'(v).
    """
    df = df.copy()

    mask1 = (df["sent_id"] == sent_id) & (df["id"] == id1)
    mask2 = (df["sent_id"] == sent_id) & (df["id"] == id2)

    if not mask1.any() or not mask2.any():
        return df

    idx1 = df[mask1].index[0]
    idx2 = df[mask2].index[0]

    # Step 1: swap attributes
    cols_to_swap = [c for c in df.columns if c not in ("sent_id", "id", "head")]
    temp = df.loc[idx1, cols_to_swap].copy()
    df.loc[idx1, cols_to_swap] = df.loc[idx2, cols_to_swap].values
    df.loc[idx2, cols_to_swap] = temp.values

    # Step 2: redirect arcs
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
    df.loc[dep_idx,  "head"] = boss_head

    for idx in df[sent_mask].index:
        if idx not in (idx1, idx2):
            if df.loc[idx, "head"] == boss_id:
                df.loc[idx, "head"] = dep_id
            elif df.loc[idx, "head"] == dep_id:
                df.loc[idx, "head"] = boss_id

    return df


def apply_all_swaps(
    df: pd.DataFrame,
    pairs: list[tuple],
) -> pd.DataFrame:
    """
    Apply all swap pairs efficiently using numpy arrays.

    Complexity: O(n_pairs × avg_sentence_length) instead of
    O(n_pairs × n_rows) from repeated df.copy() calls.

    Pairs within each sentence are applied sequentially (order matters
    for intersecting pairs, see Section 3.4).
    """
    import numpy as np
    from collections import defaultdict

    df = df.copy().reset_index(drop=True)

    # Build O(1) lookups
    row_of: dict = {}          # (sent_id, token_id) -> row index
    sent_rows: dict = {}       # sent_id -> np.array of row indices

    _sent_id_col = df["sent_id"].to_numpy()
    _id_col      = df["id"].to_numpy()

    tmp: dict = defaultdict(list)
    for i, (sid, tid) in enumerate(zip(_sent_id_col, _id_col)):
        row_of[(sid, tid)] = i
        tmp[sid].append(i)
    sent_rows = {sid: np.array(idxs) for sid, idxs in tmp.items()}

    # Extract numpy arrays (no pandas overhead inside the loop)
    attr_cols = [c for c in df.columns if c not in ("sent_id", "id", "head")]
    attr = df[attr_cols].to_numpy(dtype=object, copy=True)
    head = df["head"].to_numpy(copy=True)

    # Group pairs by sentence
    pairs_by_sent: dict = defaultdict(list)
    for sent_id, id1, id2 in pairs:
        pairs_by_sent[sent_id].append((id1, id2))

    for sent_id, sent_pairs in pairs_by_sent.items():
        s_rows = sent_rows[sent_id]  # row indices for this sentence

        for id1, id2 in sent_pairs:
            i1 = row_of.get((sent_id, id1))
            i2 = row_of.get((sent_id, id2))
            if i1 is None or i2 is None:
                continue

            # Step 1: swap attributes (numpy fancy index swap)
            attr[[i1, i2]] = attr[[i2, i1]]

            h1, h2 = int(head[i1]), int(head[i2])

            if h1 == id2:
                boss_id, dep_id = id2, id1
                boss_i,  dep_i  = i2, i1
                boss_head = h2
            elif h2 == id1:
                boss_id, dep_id = id1, id2
                boss_i,  dep_i  = i1, i2
                boss_head = h1
            else:
                continue

            # Step 2: redirect arcs
            head[boss_i] = dep_id
            head[dep_i]  = boss_head

            # Vectorized redirect for all other tokens in the sentence
            other = s_rows[(s_rows != i1) & (s_rows != i2)]
            head[other[head[other] == boss_id]] = dep_id
            head[other[head[other] == dep_id]]  = boss_id

    df[attr_cols] = attr
    df["head"]    = head
    return df


# ---------------------------------------------------------------------------
# Section 4: Edge-based comparison (auxiliary, read-only)
# ---------------------------------------------------------------------------

def _build_skeleton(df: pd.DataFrame) -> dict[str, set[frozenset]]:
    """
    Build per-sentence undirected skeletons S(T).

    S(T) = { {h(v), v} | v ∈ V \\ {0} }

    Returns dict: sent_id → set of frozenset({head, id}).
    """
    skeletons: dict = {}
    for sent_id, group in df.groupby("sent_id", sort=False):
        edges = set()
        for _, row in group.iterrows():
            h = int(row["head"])
            v = int(row["id"])
            if h != 0:
                edges.add(frozenset({h, v}))
        skeletons[sent_id] = edges
    return skeletons


def classify_edges(str_df: pd.DataFrame, ud_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify all edges across both corpora (Section 4.2, steps 1-2).

    Returns a DataFrame with columns:
      sent_id, id_i, id_j, type

    where type ∈ {"matched", "str_only", "ud_only"}.

    Formal property (Section 4.4): |D_str| == |D_ud| for every sentence.
    """
    S_str = _build_skeleton(str_df)
    S_ud  = _build_skeleton(ud_df)

    all_sent_ids = set(S_str) | set(S_ud)
    rows = []

    for sent_id in all_sent_ids:
        s = S_str.get(sent_id, set())
        u = S_ud.get(sent_id, set())

        for edge in s & u:
            i, j = sorted(edge)
            rows.append({"sent_id": sent_id, "id_i": i, "id_j": j, "type": "matched"})
        for edge in s - u:
            i, j = sorted(edge)
            rows.append({"sent_id": sent_id, "id_i": i, "id_j": j, "type": "str_only"})
        for edge in u - s:
            i, j = sorted(edge)
            rows.append({"sent_id": sent_id, "id_i": i, "id_j": j, "type": "ud_only"})

    return pd.DataFrame(rows, columns=["sent_id", "id_i", "id_j", "type"])


def compare_matched_edges(
    str_df: pd.DataFrame,
    ud_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    For each matched edge {i, j} ∈ M = S_str ∩ S_ud, find the dependent
    in each corpus and compare deprels (Section 4.2, step 3).

    Returns a DataFrame with columns:
      sent_id   — sentence identifier
      id_i      — min(i, j) of the edge
      id_j      — max(i, j)
      dep_str   — token id of the dependent in str
      dep_ud    — token id of the dependent in ud
      deprel_str — deprel label in str (at dep_str)
      deprel_ud  — deprel label in ud (at dep_ud)
      mirrored   — True when dep_str ≠ dep_ud (opposite edge orientation)
    """
    str_idx = str_df.set_index(["sent_id", "id"])
    ud_idx  = ud_df.set_index(["sent_id", "id"])

    S_str = _build_skeleton(str_df)
    S_ud  = _build_skeleton(ud_df)

    rows = []
    for sent_id in set(S_str) & set(S_ud):
        M = S_str[sent_id] & S_ud[sent_id]

        for edge in M:
            i, j = sorted(edge)

            h_str_i = int(str_idx.loc[(sent_id, i), "head"])
            dep_str = i if h_str_i == j else j

            h_ud_i = int(ud_idx.loc[(sent_id, i), "head"])
            dep_ud = i if h_ud_i == j else j

            rows.append({
                "sent_id":    sent_id,
                "id_i":       i,
                "id_j":       j,
                "dep_str":    dep_str,
                "dep_ud":     dep_ud,
                "deprel_str": str_idx.loc[(sent_id, dep_str), "deprel"],
                "deprel_ud":  ud_idx.loc[(sent_id, dep_ud),  "deprel"],
                "mirrored":   dep_str != dep_ud,
            })

    return pd.DataFrame(rows, columns=[
        "sent_id", "id_i", "id_j",
        "dep_str", "dep_ud",
        "deprel_str", "deprel_ud",
        "mirrored",
    ])
