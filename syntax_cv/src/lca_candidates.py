"""
lca_candidates.py — extended-mode candidate generation via Cross-Tree LCA.

Context
-------
In strict mode (see `edge_correspondence.build_edge_correspondence`),
a UD edge e_ud = {h_ud(v), v} is matched only against STR edges with
identical endpoints. Cases where the two trees regroup the same syntactic
material differently — "без обуви" prepositional group, conjunction
chains, auxiliary verb attachments — have no exact candidate in STR
and therefore stay unresolved.

To cover such cases without mutating either tree, we generate
non-exact candidates by walking both trees upward from v. If we can
identify a common ancestor node and a pair of arcs (one in STR, one
in UD) that both contribute to the path v → LCA, we can propose the
STR arc as a correspondence candidate for e_ud.

The candidate pool is kept small and principled: only arcs lying on
the v → LCA path in STR are proposed, and only for tokens where the
LCA is shallow (≤ `max_path_len`, default 3). This directly answers
the supervisor's concern about unbounded path length hiding structural
incompatibilities: we keep extended mode explicitly local.

No tree modifications are performed anywhere in this module.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from syntax_cv.src.edge_correspondence import Edge, TokenID, SentID


# ---------------------------------------------------------------------------
# LCA primitives (duplicated from lca_analysis.py so this module is
# self-contained; the two copies are kept in sync manually and should
# be factored out if/when lca_analysis.py is rewritten).
# ---------------------------------------------------------------------------

def _ct_lca(v: TokenID, h_str: dict, h_ud: dict) -> TokenID:
    """
    Cross-Tree LCA: nearest proper common ancestor of v that lies on v's
    path to root in BOTH trees simultaneously.

    Returns 0 when v is root in UD (no proper ancestors).
    """
    ancestors_ud: set[TokenID] = set()
    u = h_ud.get(v, 0)
    while u != 0:
        ancestors_ud.add(u)
        u = h_ud.get(u, 0)

    if not ancestors_ud:
        return 0

    u = v
    while u != 0:
        if u in ancestors_ud:
            return u
        u = h_str.get(u, 0)

    return 0


def _path_edges_to_ancestor(
    v: TokenID,
    h: dict,
    a: TokenID,
    *,
    max_len: int,
) -> list[Edge]:
    """
    Undirected edges on the path v → ... → a in tree with head map `h`.

    Stops when either (a) the ancestor `a` is reached, or (b) the path
    length exceeds `max_len`, or (c) the root is reached before `a`
    (which means a does not lie on v's path to root in this tree — the
    path is discarded and [] is returned).
    """
    edges: list[Edge] = []
    u = v
    while u != a and u != 0 and len(edges) <= max_len:
        parent = h.get(u, 0)
        if parent == 0:
            return []  # a is not on the path
        edges.append(frozenset({u, parent}))
        u = parent
    if u != a:
        return []
    return edges


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_lca_candidates(
    str_df: pd.DataFrame,
    ud_df: pd.DataFrame,
    *,
    max_path_len: int = 3,
) -> dict[SentID, dict[Edge, set[Edge]]]:
    """
    Build LCA-derived candidate edges for extended-mode edge correspondence.

    Returns a nested dict compatible with
    `edge_correspondence.build_edge_correspondence(extra_candidates=...)`:

        sent_id -> { e_ud -> set of e_str candidates }

    For each non-root UD token v:

      1. Compute a = CrossTreeLCA(v; T_str, T_ud).
         If a == 0 (no common ancestor) — skip.
      2. Compute path edges P_ud(v → a) in T_ud and P_str(v → a) in T_str,
         bounded by max_path_len.
      3. For every UD edge e_ud ∈ P_ud, propose the STR edge e_str adjacent
         to the same endpoint on the STR path. Specifically:

             ends(e_ud) = {x, parent_ud(x)}   — take x (the lower endpoint)
             find the edge in P_str whose lower endpoint is x.

         This pairs "the UD arc entering x" with "the STR arc entering x"
         on the shared subtree. When parent_str(x) ≠ parent_ud(x) the two
         arcs have different upper endpoints, so the candidate is a
         genuinely restructured match.

    The resulting candidate pool is injected into extended-mode CP. Exact
    same-endpoint candidates are handled in strict mode automatically;
    LCA candidates only matter when parent_str(x) ≠ parent_ud(x) and the
    exact-skeleton edge does not exist.

    Parameters
    ----------
    max_path_len : upper bound on |P_ud| and |P_str|. The supervisor's
        observation is that 94%+ of tokens have max(|pi_str|, |pi_ud|) ≤ 2;
        a default of 3 covers the 98%+ threshold without drifting into
        speculative long-range matchings.
    """
    str_df = str_df.reset_index(drop=False) if str_df.index.names != [None] else str_df
    ud_df = ud_df.reset_index(drop=False) if ud_df.index.names != [None] else ud_df

    result: dict[SentID, dict[Edge, set[Edge]]] = {}

    merged = str_df[["sent_id", "id", "head"]].merge(
        ud_df[["sent_id", "id", "head"]],
        on=["sent_id", "id"],
        suffixes=("_str", "_ud"),
    )

    for sent_id, grp in merged.groupby("sent_id", sort=False):
        h_str = {int(r["id"]): int(r["head_str"]) for _, r in grp.iterrows()}
        h_ud = {int(r["id"]): int(r["head_ud"]) for _, r in grp.iterrows()}

        sent_out: dict[Edge, set[Edge]] = {}

        for v, h_v_ud in h_ud.items():
            if h_v_ud == 0:
                continue

            a = _ct_lca(v, h_str, h_ud)
            if a == 0:
                continue
            if a == h_v_ud and h_str.get(v, 0) == h_v_ud:
                # Heads already match → strict mode covers it, skip.
                continue

            p_ud = _path_edges_to_ancestor(v, h_ud, a, max_len=max_path_len)
            p_str = _path_edges_to_ancestor(v, h_str, a, max_len=max_path_len)
            if not p_ud or not p_str:
                continue

            # Map every UD path edge by its lower endpoint x (the endpoint
            # that is child of the other within T_ud), and lookup an STR
            # path edge by the same x.
            str_edge_by_lower: dict[TokenID, Edge] = {}
            for e in p_str:
                x_candidates = [u for u in e if h_str.get(u) in e]
                if len(x_candidates) != 1:
                    # malformed — skip this candidate
                    continue
                str_edge_by_lower[x_candidates[0]] = e

            for e_ud in p_ud:
                x_candidates = [u for u in e_ud if h_ud.get(u) in e_ud]
                if len(x_candidates) != 1:
                    continue
                x = x_candidates[0]
                e_str = str_edge_by_lower.get(x)
                if e_str is None:
                    continue
                if e_str == e_ud:
                    # exact match already; strict mode owns it
                    continue
                sent_out.setdefault(e_ud, set()).add(e_str)

        if sent_out:
            result[sent_id] = sent_out

    return result