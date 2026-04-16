"""
Alignment of SynTagRus and UD dependency annotations.

Two approaches (see method.md):

  1. Swap-based alignment (Section 2) — primary approach.
     Detects mirrored edges and swaps str data so both corpora share
     the same arc orientation, enabling correct token-level comparison.

     Two pair-finding strategies (see swap.md, Section 4.1):

     a. find_pairs_to_swap()   — merge-based; finds all mirrored pairs
        simultaneously on the current STR tree.  For intersecting pairs
        (pairs sharing a token, e.g. a reversed chain A→B→C) the result
        depends on DataFrame iteration order, which is not guaranteed.

     b. find_pairs_to_swap_cp() + resolve_edge_matching()  — CP-based;
        uses constraint propagation (arc consistency) over UD↔STR edge
        candidates built by undirected endpoint equality, with fallback to
        all currently free STR edges when exact-end candidates are empty
        (swap.md, Sections 3-4).  This removes DataFrame-order dependence
        for intersecting/restructuring cases before swap extraction.

  2. Edge-based comparison (Section 4) — auxiliary approach.
     Read-only. Classifies edges into matched (M), str-only (D_str),
     ud-only (D_ud) and compares deprels on matched edges directly.
"""

import hashlib

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


def resolve_edge_matching(
    candidates: dict,
    all_str_edges=None,
    *,
    return_unresolved: bool = False,
) -> dict | tuple[dict, list]:
    """
    Arc consistency constraint propagation over bipartite edge matching.

    Resolves UD→STR edge assignments by the method of elimination described
    in swap.md (Sections 3-4):

      1. For each unresolved UD edge, compute available candidates as:
         (exact candidates) \ used_str.
      2. If the set is empty, fallback to all free STR edges:
         all_str_edges \ used_str.
      3. If exactly one candidate remains, fix the assignment immediately.
      4. Repeat until no further fixation is possible (stable state).

    The function is monotonic: fixed STR edges are never released, so candidate
    pools for unresolved UD edges can only shrink across iterations.

    Parameters
    ----------
    candidates : dict mapping each UD edge (frozenset of two token IDs) to
                 its exact STR candidates (set of frozensets), typically built
                 from endpoint equality.
    all_str_edges : optional iterable of all STR edges in the local zone.
                    Used for fallback when exact candidates are empty.
                    If omitted, inferred as the union of all candidate pools.
    return_unresolved : when True, also return unresolved UD edges left after
                        propagation.

    Returns
    -------
    fixed : dict, UD edge → matched STR edge, for resolved assignments.
            If return_unresolved=True, returns tuple:
            (fixed, unresolved_ud_edges_in_input_order).
    """
    fixed: dict = {}
    unresolved = list(candidates.keys())

    if all_str_edges is None:
        all_free_pool = set()
        for pool in candidates.values():
            all_free_pool |= set(pool)
    else:
        all_free_pool = set(all_str_edges)

    made_progress = True
    while made_progress:
        made_progress = False
        used_str = set(fixed.values())
        next_unresolved = []

        for e_ud in unresolved:
            exact_pool = candidates.get(e_ud, set())
            available = exact_pool - used_str

            # swap.md fallback: if exact candidates are empty, open all free STR edges
            if not available:
                available = all_free_pool - used_str

            if len(available) == 1:
                chosen = next(iter(available))
                fixed[e_ud] = chosen
                used_str.add(chosen)
                made_progress = True
            else:
                next_unresolved.append(e_ud)

        unresolved = next_unresolved

    if return_unresolved:
        return fixed, unresolved
    return fixed


def _build_directed_skeleton_by_sent(df: pd.DataFrame) -> dict:
    """
    Build per-sentence undirected skeleton -> head-id mapping.

    Returns:
      sent_id -> {frozenset({h, v}): h}
    """
    out: dict = {}
    cols = df[["sent_id", "id", "head"]]
    for sent_id, grp in cols.groupby("sent_id", sort=False):
        ids = grp["id"].to_numpy(copy=False)
        heads = grp["head"].to_numpy(copy=False)
        edges: dict = {}
        for v, h in zip(ids, heads):
            h = int(h)
            if h == 0:
                continue
            v = int(v)
            edges[frozenset({h, v})] = h
        out[sent_id] = edges
    return out


def _find_pairs_to_swap_cp_from_skeletons(
    str_skel_by_sent: dict,
    ud_skel_by_sent: dict,
    extra_candidates=None,
    *,
    return_diagnostics: bool = False,
) -> list[tuple] | tuple[list[tuple], dict]:
    """
    Internal CP pair extraction from prebuilt sentence skeletons.

    This helper allows recursive alignment to reuse immutable UD skeletons
    between passes and avoid repeated DataFrame scans.
    """
    pairs: list = []
    seen: set = set()
    diagnostics = {
        "sentences_total": 0,
        "sentences_with_unresolved": 0,
        "total_ud_edges": 0,
        "resolved_ud_edges": 0,
        "unresolved_ud_edges": 0,
        "unresolved_sent_ids": [],
        "unresolved_by_sentence": {},
    }

    for sent_id, str_skel in str_skel_by_sent.items():
        ud_skel = ud_skel_by_sent.get(sent_id)
        if ud_skel is None:
            continue

        all_str_edges = set(str_skel.keys())
        diagnostics["sentences_total"] += 1

        # Default exact candidates by endpoint equality (swap.md Section 3).
        # Since edges are undirected frozensets, equality means same endpoints.
        candidates = {e_ud: ({e_ud} if e_ud in str_skel else set()) for e_ud in ud_skel}

        # Merge extra candidates (e.g., from LCA analysis)
        if extra_candidates and sent_id in extra_candidates:
            for e_ud, extra in extra_candidates[sent_id].items():
                if e_ud in candidates:
                    candidates[e_ud] |= extra
                else:
                    candidates[e_ud] = set(extra)

        # Constraint propagation with swap.md fallback on empty exact pools
        fixed, unresolved = resolve_edge_matching(
            candidates,
            all_str_edges=all_str_edges,
            return_unresolved=True,
        )
        diagnostics["total_ud_edges"] += len(ud_skel)
        diagnostics["resolved_ud_edges"] += len(fixed)
        diagnostics["unresolved_ud_edges"] += len(unresolved)
        if unresolved:
            diagnostics["sentences_with_unresolved"] += 1
            diagnostics["unresolved_sent_ids"].append(sent_id)
            diagnostics["unresolved_by_sentence"][sent_id] = len(unresolved)

        # Extract mirrored pairs from same-skeleton fixed matchings
        for e_ud, e_str in fixed.items():
            if e_ud != e_str:
                continue  # different skeleton — not a direct-swap candidate
            head_str = str_skel.get(e_str)
            head_ud = ud_skel.get(e_ud)
            if head_str is None or head_ud is None:
                continue
            if head_str != head_ud:  # arc directions differ → mirrored
                i, j = sorted(e_ud)
                key = (sent_id, i, j)
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)

    if return_diagnostics:
        return pairs, diagnostics
    return pairs


def find_pairs_to_swap_cp(
    str_df: pd.DataFrame,
    ud_df: pd.DataFrame,
    extra_candidates=None,
    *,
    return_diagnostics: bool = False,
) -> list[tuple] | tuple[list[tuple], dict]:
    """
    Find mirrored pairs via constraint propagation (Section 2.1, CP variant).

    Replaces the merge-based find_pairs_to_swap() with an explicit bipartite
    edge-matching stage that resolves ambiguities before testing for mirrored
    directions. This directly addresses the ordering problem for intersecting
    pairs described in swap.md Section 4.1.

    Candidacy criterion (default; swap.md Sections 3-4)
    ---------------------------------------------------
    For each UD edge e_ud, exact STR candidates are edges with identical
    undirected endpoints:

      C(e_ud) = {e_str in E_str : ends(e_str) = ends(e_ud)}

    If the exact pool is empty at a propagation step, candidate set is opened
    to all free STR edges not yet used in fixed assignments.

    Mirrored-pair extraction
    ------------------------
    From the fixed UD→STR matchings, a pair (i, j) is emitted when:
      • e_ud == e_str  (same undirected skeleton, i.e. same two token IDs), AND
      • head_str ≠ head_ud  (opposite arc directions → mirrored).

    Fixed matchings where e_ud ≠ e_str (different skeletons) are skipped here;
    they correspond to non-trivial restructurings and require separate handling.

    Parameters
    ----------
    str_df, ud_df    : DataFrames with columns [sent_id, id, head, ...]
    extra_candidates : optional dict  sent_id → {e_ud → set(e_str)},
                       merged into the default candidates before propagation.
                       Use this to inject LCA-based or other non-skeleton
                       restructuring candidates without modifying this function.

    Returns
    -------
    list of (sent_id, id_min, id_max) — same format as find_pairs_to_swap(),
    in CP resolution order within each sentence.

    If return_diagnostics=True, also returns a dict with aggregate CP status:
      - sentences_total
      - sentences_with_unresolved
      - total_ud_edges
      - resolved_ud_edges
      - unresolved_ud_edges
      - unresolved_sent_ids
      - unresolved_by_sentence
    """
    str_skel_by_sent = _build_directed_skeleton_by_sent(str_df)
    ud_skel_by_sent = _build_directed_skeleton_by_sent(ud_df)

    return _find_pairs_to_swap_cp_from_skeletons(
        str_skel_by_sent,
        ud_skel_by_sent,
        extra_candidates=extra_candidates,
        return_diagnostics=return_diagnostics,
    )


def swap_rows_in_df(
    df:      pd.DataFrame,
    sent_id,
    id1:     int,
    id2:     int,
) -> pd.DataFrame:
    """
    Apply swap(b, d) to df (Section 2.2).

    Step 1: exchange all attributes except sent_id, id, head.
    Step 2: rewire only the swapped pair itself:
            former boss becomes dependent of former dependent, and
            former dependent inherits former boss head.

    Other sentence tokens keep their head IDs unchanged; ambiguity on
    adjacent links is resolved by iterative pair selection (method of
    exclusion in find_pairs_to_swap_cp/resolve_edge_matching).
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

    # Step 2: rewire swapped pair
    head1 = df.loc[idx1, "head"]
    head2 = df.loc[idx2, "head"]

    if head1 == id2:
        dep_id = id1
        boss_idx, dep_idx = idx2, idx1
        boss_head = head2
    elif head2 == id1:
        dep_id = id2
        boss_idx, dep_idx = idx1, idx2
        boss_head = head1
    else:
        return df

    df.loc[boss_idx, "head"] = dep_id
    df.loc[dep_idx,  "head"] = boss_head

    return df


def apply_all_swaps(
    df: pd.DataFrame,
    pairs: list[tuple],
) -> pd.DataFrame:
    """
    Apply all swap pairs efficiently using numpy arrays.

    Complexity: O(n_rows + n_pairs) with vectorized arrays and O(1) row lookups.

    Pairs within each sentence are applied sequentially (order matters
    for intersecting pairs, see Section 3.4).

    Rewiring is local to each swapped pair; heads of all other tokens in
    the sentence are left intact.
    """
    from collections import defaultdict

    df = df.copy().reset_index(drop=True)

    # Build O(1) lookups
    row_of: dict = {}          # (sent_id, token_id) -> row index

    _sent_id_col = df["sent_id"].to_numpy()
    _id_col      = df["id"].to_numpy()

    for i, (sid, tid) in enumerate(zip(_sent_id_col, _id_col)):
        row_of[(sid, tid)] = i

    # Extract numpy arrays (no pandas overhead inside the loop)
    attr_cols = [c for c in df.columns if c not in ("sent_id", "id", "head")]
    attr = df[attr_cols].to_numpy(dtype=object, copy=True)
    head = df["head"].to_numpy(copy=True)

    # Group pairs by sentence
    pairs_by_sent: dict = defaultdict(list)
    for sent_id, id1, id2 in pairs:
        pairs_by_sent[sent_id].append((id1, id2))

    for sent_id, sent_pairs in pairs_by_sent.items():
        for id1, id2 in sent_pairs:
            i1 = row_of.get((sent_id, id1))
            i2 = row_of.get((sent_id, id2))
            if i1 is None or i2 is None:
                continue

            # Step 1: swap attributes (numpy fancy index swap)
            attr[[i1, i2]] = attr[[i2, i1]]

            h1, h2 = int(head[i1]), int(head[i2])

            if h1 == id2:
                dep_id = id1
                boss_i,  dep_i  = i2, i1
                boss_head = h2
            elif h2 == id1:
                dep_id = id2
                boss_i,  dep_i  = i1, i2
                boss_head = h1
            else:
                continue

            # Step 2: rewire swapped pair
            head[boss_i] = dep_id
            head[dep_i]  = boss_head

    df[attr_cols] = attr
    df["head"]    = head
    return df


def _head_signature(df: pd.DataFrame) -> str:
    """
    Compact hash of head column for cycle detection between recursive passes.
    """
    payload = df["head"].to_numpy(copy=False).tobytes()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def build_recursive_swap_plan(
    str_df: pd.DataFrame,
    ud_df: pd.DataFrame,
    max_passes: int = 20,
    stop_on_cycle: bool = True,
) -> list[list[tuple]]:
    """
    Build recursive swap plan with merge-based pair finder.

    Pass k:
      1) find mirrored pairs on current STR tree vs fixed UD tree;
      2) apply all swaps from this pass;
      3) continue until no pairs remain or max_passes reached.

    Returns:
      list of passes, where each pass is list[(sent_id, id1, id2)].
    """
    current = str_df.copy()
    plan: list[list[tuple]] = []

    seen_states: set[str] = {_head_signature(current)}

    for _ in range(max_passes):
        pairs = find_pairs_to_swap(current, ud_df)
        if not pairs:
            break

        plan.append(pairs)
        current = apply_all_swaps(current, pairs)

        if stop_on_cycle:
            sig = _head_signature(current)
            if sig in seen_states:
                break
            seen_states.add(sig)

    return plan


def build_recursive_swap_plan_cp(
    str_df: pd.DataFrame,
    ud_df: pd.DataFrame,
    max_passes: int = 20,
    stop_on_cycle: bool = True,
    extra_candidates=None,
    *,
    return_diagnostics: bool = False,
) -> list[list[tuple]] | tuple[list[list[tuple]], dict]:
    """
    Build recursive swap plan with CP pair finder.

    At each pass, recompute CP-resolved mirrored pairs on the current STR tree.
    This implements recursive "method of exclusion":
    keep fixing unambiguous choices until stabilization, then repeat on the
    updated tree while new mirrored pairs continue to appear.
    """
    current = str_df.copy()
    plan: list[list[tuple]] = []
    pass_diagnostics: list[dict] = []
    ud_skel_by_sent = _build_directed_skeleton_by_sent(ud_df)

    seen_states: set[str] = {_head_signature(current)}

    for _ in range(max_passes):
        current_skel_by_sent = _build_directed_skeleton_by_sent(current)
        pairs, diag = _find_pairs_to_swap_cp_from_skeletons(
            current_skel_by_sent,
            ud_skel_by_sent,
            extra_candidates=extra_candidates,
            return_diagnostics=True,
        )
        pass_diagnostics.append(diag)
        if not pairs:
            break

        plan.append(pairs)
        current = apply_all_swaps(current, pairs)

        if stop_on_cycle:
            sig = _head_signature(current)
            if sig in seen_states:
                break
            seen_states.add(sig)

    if return_diagnostics:
        summary = {
            "passes": len(plan),
            "pairs_total": sum(len(p) for p in plan),
            "pass_diagnostics": pass_diagnostics,
            "any_unresolved": any(d["unresolved_ud_edges"] > 0 for d in pass_diagnostics),
            "last_unresolved_ud_edges": pass_diagnostics[-1]["unresolved_ud_edges"] if pass_diagnostics else 0,
            "last_sentences_with_unresolved": pass_diagnostics[-1]["sentences_with_unresolved"] if pass_diagnostics else 0,
            "last_unresolved_sent_ids": pass_diagnostics[-1]["unresolved_sent_ids"] if pass_diagnostics else [],
            "last_unresolved_by_sentence": pass_diagnostics[-1]["unresolved_by_sentence"] if pass_diagnostics else {},
        }
        return plan, summary

    return plan


def apply_swap_plan(df: pd.DataFrame, plan: list[list[tuple]]) -> pd.DataFrame:
    """
    Apply recursive swap plan produced by build_recursive_swap_plan().
    """
    out = df.copy()
    for pass_pairs in plan:
        out = apply_all_swaps(out, pass_pairs)
    return out


def align_recursively(
    str_df: pd.DataFrame,
    ud_df: pd.DataFrame,
    max_passes: int = 20,
    stop_on_cycle: bool = True,
    method: str = "cp",
    return_plan: bool = False,
):
    """
    Recursively align STR against UD by repeated swap passes.

    Returns:
      aligned_str_df
    or
      (aligned_str_df, plan) if return_plan=True.
    """
    if method == "cp":
        plan = build_recursive_swap_plan_cp(
            str_df=str_df,
            ud_df=ud_df,
            max_passes=max_passes,
            stop_on_cycle=stop_on_cycle,
        )
    elif method == "merge":
        plan = build_recursive_swap_plan(
            str_df=str_df,
            ud_df=ud_df,
            max_passes=max_passes,
            stop_on_cycle=stop_on_cycle,
        )
    else:
        raise ValueError("method must be 'cp' or 'merge'")

    aligned = apply_swap_plan(str_df, plan)

    if return_plan:
        return aligned, plan
    return aligned


def get_unresolved_alignment_sentences(
    str_df: pd.DataFrame,
    ud_df: pd.DataFrame,
    max_passes: int = 20,
    stop_on_cycle: bool = True,
    extra_candidates=None,
    with_text: bool = True,
) -> pd.DataFrame:
    """
    Return sentences where recursive CP alignment remains ambiguous.

    Ambiguous means that after the last recursive pass there are UD edges
    that still have multiple possible STR matches (no unique fixation by
    elimination).
    """
    _, summary = build_recursive_swap_plan_cp(
        str_df=str_df,
        ud_df=ud_df,
        max_passes=max_passes,
        stop_on_cycle=stop_on_cycle,
        extra_candidates=extra_candidates,
        return_diagnostics=True,
    )

    unresolved_by_sentence = summary.get("last_unresolved_by_sentence", {})
    if not unresolved_by_sentence:
        cols = ["sent_id", "unresolved_ud_edges"]
        if with_text:
            cols.append("text")
        return pd.DataFrame(columns=cols)

    rows = [
        {"sent_id": sent_id, "unresolved_ud_edges": n}
        for sent_id, n in unresolved_by_sentence.items()
    ]
    out = pd.DataFrame(rows).sort_values(
        ["unresolved_ud_edges", "sent_id"],
        ascending=[False, True],
    )

    if with_text and "text" in str_df.columns:
        sent_text = str_df.groupby("sent_id", sort=False)["text"].first()
        out["text"] = out["sent_id"].map(sent_text)

    return out.reset_index(drop=True)


def inspect_unresolved_alignment_sentence(
    str_df: pd.DataFrame,
    ud_df: pd.DataFrame,
    sent_id: str,
    max_passes: int = 20,
    stop_on_cycle: bool = True,
    extra_candidates=None,
) -> pd.DataFrame:
    """
    Inspect token-level ambiguity for one sentence after recursive CP alignment.

    Returns one row per unresolved UD edge x available STR candidate with token
    IDs/forms and directed heads in both trees after applying the recursive
    swap plan to STR.
    """
    str_sent = str_df[str_df["sent_id"] == sent_id].copy()
    ud_sent = ud_df[ud_df["sent_id"] == sent_id].copy()
    if str_sent.empty or ud_sent.empty:
        raise ValueError(f"sent_id={sent_id!r} not found in one of the inputs")

    plan = build_recursive_swap_plan_cp(
        str_df=str_sent,
        ud_df=ud_sent,
        max_passes=max_passes,
        stop_on_cycle=stop_on_cycle,
        extra_candidates=extra_candidates,
    )
    str_aligned = apply_swap_plan(str_sent, plan)

    str_skel_by_sent = _build_directed_skeleton_by_sent(str_aligned)
    ud_skel_by_sent = _build_directed_skeleton_by_sent(ud_sent)
    str_skel = str_skel_by_sent.get(sent_id, {})
    ud_skel = ud_skel_by_sent.get(sent_id, {})
    all_str_edges = set(str_skel.keys())

    candidates = {e_ud: ({e_ud} if e_ud in str_skel else set()) for e_ud in ud_skel}
    if extra_candidates and sent_id in extra_candidates:
        for e_ud, extra in extra_candidates[sent_id].items():
            if e_ud in candidates:
                candidates[e_ud] |= extra
            else:
                candidates[e_ud] = set(extra)

    fixed, unresolved = resolve_edge_matching(
        candidates,
        all_str_edges=all_str_edges,
        return_unresolved=True,
    )
    used_str = set(fixed.values())

    columns = [
        "sent_id",
        "plan_passes",
        "plan_pairs_total",
        "ud_i",
        "ud_form_i",
        "ud_j",
        "ud_form_j",
        "ud_head",
        "ud_dep",
        "str_i",
        "str_form_i",
        "str_form_i_original",
        "str_j",
        "str_form_j",
        "str_form_j_original",
        "str_head",
        "str_dep",
    ]
    if not unresolved:
        return pd.DataFrame(columns=columns)

    ud_id2form = {}
    if "form" in ud_sent.columns:
        ud_id2form = dict(zip(ud_sent["id"].astype(int), ud_sent["form"]))

    str_id2form_aligned = {}
    if "form" in str_aligned.columns:
        str_id2form_aligned = dict(zip(str_aligned["id"].astype(int), str_aligned["form"]))

    str_id2form_original = {}
    if "form" in str_sent.columns:
        str_id2form_original = dict(zip(str_sent["id"].astype(int), str_sent["form"]))

    plan_pairs_total = sum(len(pass_pairs) for pass_pairs in plan)
    rows = []
    for e_ud in unresolved:
        ud_i, ud_j = sorted(e_ud)
        ud_head = ud_skel.get(e_ud)
        ud_dep = None
        if ud_head is not None:
            ud_dep = ud_j if ud_head == ud_i else ud_i

        exact_available = candidates.get(e_ud, set()) - used_str
        available_str = exact_available if exact_available else (all_str_edges - used_str)
        for e_str in sorted(available_str, key=lambda e: tuple(sorted(e))):
            str_i, str_j = sorted(e_str)
            str_head = str_skel.get(e_str)
            str_dep = None
            if str_head is not None:
                str_dep = str_j if str_head == str_i else str_i

            rows.append({
                "sent_id": sent_id,
                "plan_passes": len(plan),
                "plan_pairs_total": plan_pairs_total,
                "ud_i": ud_i,
                "ud_form_i": ud_id2form.get(ud_i),
                "ud_j": ud_j,
                "ud_form_j": ud_id2form.get(ud_j),
                "ud_head": ud_head,
                "ud_dep": ud_dep,
                "str_i": str_i,
                "str_form_i": str_id2form_aligned.get(str_i),
                "str_form_i_original": str_id2form_original.get(str_i),
                "str_j": str_j,
                "str_form_j": str_id2form_aligned.get(str_j),
                "str_form_j_original": str_id2form_original.get(str_j),
                "str_head": str_head,
                "str_dep": str_dep,
            })

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["ud_i", "ud_j", "str_i", "str_j"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section 4: Edge-based comparison (auxiliary, read-only)
# ---------------------------------------------------------------------------

def _build_skeleton(df: pd.DataFrame) -> dict[str, set[frozenset]]:
    """
    Build per-sentence undirected skeletons S(T).

    S(T) = { {h(v), v} | v ∈ V \\ {0} }

    Returns dict: sent_id → set of frozenset({head, id}).
    """
    directed = _build_directed_skeleton_by_sent(df)
    return {sent_id: set(edges.keys()) for sent_id, edges in directed.items()}


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
