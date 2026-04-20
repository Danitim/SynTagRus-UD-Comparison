"""
topdown_matching.py — Top-down tree matching for SynTagRus↔UD correspondence.

Algorithm (supervisor's specification, 17-Apr-26):

    0. **Lock exact edges globally**: if the same undirected edge exists in
       both trees, keep it as identity correspondence.

    1. Start from the roots of both trees — even if the root tokens differ —
       and walk downward, pairing subtrees along the way.

    At each paired node (str_node, ud_node), outgoing edges (parent → child)
    are matched in four phases:

        1. **Exact**: the undirected edge appears in both trees → identity match.
        2. **Shared child**: edges leading to the same dep-endpoint token → pair.
        3. **Singleton**: exactly one unmatched edge on each side → forced pair.
        4. **Brute force**: for ≤ MAX_BRUTE remaining edges, enumerate all
           assignments and pick the one with the highest subtree-token overlap.

    Before phase 4, an anchor pass is applied for 1-vs-many leftovers:
    if there is a unique edge pair with positive endpoint overlap, it is
    fixed greedily to avoid accidental swaps caused by subtree-size ties.

    After matching edges at a node, the algorithm recurses into each matched
    (child_str, child_ud) pair, expanding the frontier one level deeper.
    Then a second pass runs from identity node anchors (t, t) to cover
    unresolved local regions that may be disconnected from the root pair.

    The procedure is designed for sentences of moderate length (≤ 50 tokens).
    Complexity is O(n² + Σ k!) where k is the number of unmatched children
    at a node — effectively O(n²) in practice because k rarely exceeds 3.

No input DataFrames or Skeletons are mutated.  The output is a dict
e_ud → e_str, directly consumable by ``edge_correspondence``.
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Optional

from src.edge_correspondence import (
    Edge,
    SentenceSkeleton,
    TokenID,
)

MAX_BRUTE = 6  # skip exhaustive search when either side exceeds this


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_children_map(
    sk: SentenceSkeleton,
) -> dict[TokenID, list[tuple[Edge, TokenID]]]:
    """Parent → [(edge, child_token)] mapping."""
    out: dict[TokenID, list[tuple[Edge, TokenID]]] = {}
    for e in sk.edges:
        head = sk.head_of(e)
        dep = sk.dep_of(e)
        if head is not None and dep is not None:
            out.setdefault(head, []).append((e, dep))
    return out


def _root_tokens(sk: SentenceSkeleton) -> list[TokenID]:
    """Sorted list of root token ids (tokens that are never a dependent)."""
    deps = {d for d in (sk.dep_of(e) for e in sk.edges) if d is not None}
    return sorted(v for v in sk.token_ids if v not in deps)


def _subtree_tokens(
    node: TokenID,
    children: dict[TokenID, list[tuple[Edge, TokenID]]],
) -> frozenset[TokenID]:
    """All tokens in the subtree rooted at *node* (inclusive)."""
    result = {node}
    stack = [node]
    while stack:
        n = stack.pop()
        for _, child in children.get(n, []):
            if child not in result:
                result.add(child)
                stack.append(child)
    return frozenset(result)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def topdown_match(
    str_sk: SentenceSkeleton,
    ud_sk: SentenceSkeleton,
) -> dict[Edge, Edge]:
    """
    Top-down tree matching between STR and UD sentence skeletons.

    Returns a dict mapping ``e_ud → e_str`` for every resolved UD edge.
    Unresolved UD edges are simply absent from the dict.
    """
    str_ch = _build_children_map(str_sk)
    ud_ch = _build_children_map(ud_sk)

    # Pre-compute subtree token sets (used by Phase 4 scoring).
    str_sub: dict[TokenID, frozenset[TokenID]] = {
        t: _subtree_tokens(t, str_ch) for t in str_sk.token_ids
    }
    ud_sub: dict[TokenID, frozenset[TokenID]] = {
        t: _subtree_tokens(t, ud_ch) for t in ud_sk.token_ids
    }

    result: dict[Edge, Edge] = {}
    used_ud: set[Edge] = set()
    used_str: set[Edge] = set()
    mirror_role_map: dict[TokenID, TokenID] = {}

    def _assign(
        e_ud: Edge,
        e_str: Edge,
        dep_ud: TokenID,
        dep_str: TokenID,
        frontier: list[tuple[TokenID, TokenID]],
    ) -> bool:
        """Injective assignment helper."""
        if e_ud in used_ud or e_str in used_str:
            return False
        result[e_ud] = e_str
        used_ud.add(e_ud)
        used_str.add(e_str)
        frontier.append((dep_str, dep_ud))
        return True

    # Keep strict identity edges fixed globally.
    for e_ud in ud_sk.edges:
        if e_ud in str_sk.edges:
            dep_ud = ud_sk.dep_of(e_ud)
            dep_str = str_sk.dep_of(e_ud)
            if dep_ud is not None and dep_str is not None:
                _assign(e_ud, e_ud, dep_ud, dep_str, frontier=[])
                h_ud = ud_sk.head_of(e_ud)
                h_str = str_sk.head_of(e_ud)
                if h_ud is not None and h_str is not None and h_ud != h_str:
                    # Exact mirrored edge: keep role-aware endpoint mapping
                    # for tie-breaking ambiguous local assignments.
                    mirror_role_map[h_ud] = h_str
                    mirror_role_map[dep_ud] = dep_str

    # ------------------------------------------------------------------
    def _match(s_node: TokenID, u_node: TokenID) -> None:
        """Match children of *s_node* (T_str) with children of *u_node* (T_ud)."""
        s_edges = [
            (e, dep) for e, dep in str_ch.get(s_node, [])
            if e not in used_str
        ]
        u_edges = [
            (e, dep) for e, dep in ud_ch.get(u_node, [])
            if e not in used_ud
        ]
        if not u_edges or not s_edges:
            return

        frontier: list[tuple[TokenID, TokenID]] = []

        # ---- Phase 1: exact undirected edge --------------------------------
        s_avail = {e: dep for e, dep in s_edges}
        u_rem: list[tuple[Edge, TokenID]] = []

        for e_ud, dep_ud in u_edges:
            dep_str = s_avail.get(e_ud)
            if dep_str is not None:
                if _assign(e_ud, e_ud, dep_ud, dep_str, frontier):
                    s_avail.pop(e_ud, None)
            else:
                u_rem.append((e_ud, dep_ud))

        if not u_rem:
            for cs, cu in frontier:
                _match(cs, cu)
            return

        s_rem = [(e, dep) for e, dep in s_edges if e in s_avail]

        # ---- Phase 2: shared dep (child) token -----------------------------
        s_by_dep: dict[TokenID, list[tuple[Edge, TokenID]]] = {}
        for e, dep in s_rem:
            s_by_dep.setdefault(dep, []).append((e, dep))

        used_local: set[Edge] = set()
        still_u: list[tuple[Edge, TokenID]] = []

        for e_ud, dep_ud in u_rem:
            cands = [x for x in s_by_dep.get(dep_ud, []) if x[0] not in used_local]
            if len(cands) == 1:
                e_str, dep_str = cands[0]
                if _assign(e_ud, e_str, dep_ud, dep_str, frontier):
                    used_local.add(e_str)
                else:
                    still_u.append((e_ud, dep_ud))
            else:
                still_u.append((e_ud, dep_ud))

        still_s = [(e, dep) for e, dep in s_rem if e not in used_local]

        # ---- Phase 3: singleton elimination --------------------------------
        def _anchor_bonus(e_ud: Edge, e_str: Edge) -> int:
            bonus = 0
            for u in e_ud:
                mapped = mirror_role_map.get(u)
                if mapped is not None and mapped in e_str:
                    bonus += 1
            return bonus

        if len(still_s) == 1 and len(still_u) > 1:
            e_str, dep_str = still_s[0]
            overlap = [len(e_ud & e_str) for e_ud, _ in still_u]
            best = max(overlap)
            if best > 0:
                cands = [j for j, ov in enumerate(overlap) if ov == best]
                if len(cands) == 1:
                    j = cands[0]
                else:
                    anchors = [_anchor_bonus(still_u[j][0], e_str) for j in cands]
                    a_best = max(anchors)
                    if anchors.count(a_best) != 1:
                        j = -1
                    else:
                        j = cands[anchors.index(a_best)]
                if j >= 0:
                    e_ud, dep_ud = still_u[j]
                    if _assign(e_ud, e_str, dep_ud, dep_str, frontier):
                        still_u = [x for k, x in enumerate(still_u) if k != j]
                        still_s = []

        if len(still_u) == 1 and len(still_s) > 1:
            e_ud, dep_ud = still_u[0]
            overlap = [len(e_ud & e_str) for e_str, _ in still_s]
            best = max(overlap)
            if best > 0:
                cands = [j for j, ov in enumerate(overlap) if ov == best]
                if len(cands) == 1:
                    j = cands[0]
                else:
                    anchors = [_anchor_bonus(e_ud, still_s[j][0]) for j in cands]
                    a_best = max(anchors)
                    if anchors.count(a_best) != 1:
                        j = -1
                    else:
                        j = cands[anchors.index(a_best)]
                if j >= 0:
                    e_str, dep_str = still_s[j]
                    if _assign(e_ud, e_str, dep_ud, dep_str, frontier):
                        still_u = []
                        still_s = [x for k, x in enumerate(still_s) if k != j]

        if len(still_u) == 1 and len(still_s) == 1:
            e_ud, dep_ud = still_u[0]
            e_str, dep_str = still_s[0]
            if _assign(e_ud, e_str, dep_ud, dep_str, frontier):
                still_u = []
                still_s = []

        # ---- Phase 4: brute force by subtree overlap -----------------------
        if still_u and still_s:
            nu, ns = len(still_u), len(still_s)
            k = min(nu, ns)

            if nu <= MAX_BRUTE and ns <= MAX_BRUTE:
                # When nu > ns we also choose *which* UD edges to match.
                u_choices: list[tuple[int, ...]] = (
                    [tuple(range(nu))]
                    if nu <= ns
                    else list(combinations(range(nu), k))
                )

                best_score = -1
                best_anchor_score = -1
                best_u_idx: Optional[tuple[int, ...]] = None
                best_perm: Optional[tuple[int, ...]] = None

                for u_idx in u_choices:
                    for perm in permutations(range(ns), k):
                        score = 0
                        anchor_score = 0
                        for j in range(k):
                            cu = still_u[u_idx[j]][1]
                            e_ud = still_u[u_idx[j]][0]
                            e_str = still_s[perm[j]][0]
                            cs = still_s[perm[j]][1]
                            score += len(
                                ud_sub.get(cu, frozenset())
                                & str_sub.get(cs, frozenset())
                            )
                            anchor_score += _anchor_bonus(e_ud, e_str)
                        if (
                            score > best_score
                            or (score == best_score and anchor_score > best_anchor_score)
                        ):
                            best_score = score
                            best_anchor_score = anchor_score
                            best_u_idx = u_idx
                            best_perm = perm

                if best_perm is not None and best_u_idx is not None:
                    for j in range(k):
                        e_ud, dep_ud = still_u[best_u_idx[j]]
                        e_str, dep_str = still_s[best_perm[j]]
                        _assign(e_ud, e_str, dep_ud, dep_str, frontier)

        # ---- Recurse -------------------------------------------------------
        for cs, cu in frontier:
            _match(cs, cu)

    # ------------------------------------------------------------------
    # Pair roots and launch recursion
    # ------------------------------------------------------------------
    s_roots = _root_tokens(str_sk)
    u_roots = _root_tokens(ud_sk)

    if len(s_roots) == 1 and len(u_roots) == 1:
        _match(s_roots[0], u_roots[0])
    elif s_roots and u_roots:
        # Multi-root: match by token equality first, then by overlap.
        s_set, u_set = set(s_roots), set(u_roots)
        for r in sorted(s_set & u_set):
            _match(r, r)

        s_left = sorted(s_set - u_set)
        u_left = sorted(u_set - s_set)
        k = min(len(u_left), len(s_left))
        if 0 < k <= MAX_BRUTE and len(s_left) <= MAX_BRUTE:
            best_score = -1
            best_perm: Optional[tuple[int, ...]] = None
            for perm in permutations(range(len(s_left)), k):
                score = sum(
                    len(
                        ud_sub.get(u_left[i], frozenset())
                        & str_sub.get(s_left[perm[i]], frozenset())
                    )
                    for i in range(k)
                )
                if score > best_score:
                    best_score = score
                    best_perm = perm
            if best_perm is not None:
                for i in range(k):
                    _match(s_left[best_perm[i]], u_left[i])

    # Second pass: identity anchors for unresolved regions.
    common_nodes = sorted(set(str_sk.token_ids) & set(ud_sk.token_ids))
    for node in common_nodes:
        s_has_open = any(e not in used_str for e, _ in str_ch.get(node, []))
        u_has_open = any(e not in used_ud for e, _ in ud_ch.get(node, []))
        if s_has_open and u_has_open:
            _match(node, node)

    # Third pass: bridge residual UD edges that keep the same dependent token
    # but switch parent anchor along an already matched parent-parent edge.
    #
    # Typical case: UD star (u->a, u->b) vs STR chain (u->a, a->b).
    incoming_str: dict[TokenID, Edge] = {}
    for e_str in str_sk.edges:
        d = str_sk.dep_of(e_str)
        if d is not None:
            incoming_str[d] = e_str

    matched_str_edges = set(result.values())
    for e_ud in sorted(ud_sk.edges, key=lambda e: sorted(e)):
        if e_ud in used_ud:
            continue
        h_ud = ud_sk.head_of(e_ud)
        d_ud = ud_sk.dep_of(e_ud)
        if h_ud is None or d_ud is None:
            continue

        e_str = incoming_str.get(d_ud)
        if e_str is None or e_str in used_str:
            continue

        h_str = str_sk.head_of(e_str)
        if h_str is None:
            continue

        bridge = frozenset({h_ud, h_str})
        if bridge in matched_str_edges or (bridge in str_sk.edges and bridge in ud_sk.edges):
            if _assign(e_ud, e_str, d_ud, d_ud, frontier=[]):
                matched_str_edges.add(e_str)

    return result
