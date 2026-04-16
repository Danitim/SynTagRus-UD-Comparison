"""
resolve.py — Arc-consistency constraint propagation for bipartite UD↔STR
edge matching.

Extracted from the original `align_markup.py`. The only substantive change
compared to the previous implementation is the explicit `allow_open_fallback`
flag, which is now False by default. When False the solver never opens the
candidate pool to "all free STR edges"; this was the mechanism responsible
for fabricating arc-direction matchings that the supervisor flagged as
incorrect (see correspondence 16-Apr-26 about the "надо/бы" analytical
link). In the immutable-tree regime introduced by `edge_correspondence.py`
we explicitly require each candidate pool to be supplied by principled
means (exact endpoint equality in strict mode, LCA-derived edges in
extended mode).

The algorithm itself is a textbook AC-3 restricted to singleton-fixation:
repeatedly scan unresolved UD edges, intersect their candidate pool with
the set of still-unused STR edges, and fix the match whenever exactly one
candidate survives. Monotone: once an STR edge is assigned it is never
released, so the procedure terminates in O(|E_ud|²) in the worst case.
"""

from __future__ import annotations

from typing import Iterable, Optional


def resolve_edge_matching(
    candidates: dict,
    all_str_edges: Optional[Iterable] = None,
    *,
    allow_open_fallback: bool = False,
    return_unresolved: bool = False,
):
    """
    AC-3 style propagation over bipartite UD↔STR edge candidates.

    Parameters
    ----------
    candidates
        Mapping e_ud → set(e_str). Edges are frozensets of two token ids.
    all_str_edges
        Optional iterable of all STR edges in the sentence. Only consulted
        when allow_open_fallback=True; otherwise ignored.
    allow_open_fallback
        If True, an unresolved e_ud with an empty candidate pool falls
        back to "every STR edge not yet used". This is the legacy
        behaviour from `align_markup.py`. DEFAULT FALSE — we prefer
        honest `unresolved` outcomes over speculative matchings.
    return_unresolved
        If True, return `(fixed, unresolved)` where `unresolved` is the
        list of e_ud whose candidate pool collapsed to {} or >1 and never
        became singleton during propagation.

    Returns
    -------
    fixed : dict e_ud → e_str
    unresolved : list[e_ud]   (only if return_unresolved)
    """
    fixed: dict = {}
    unresolved = list(candidates.keys())

    if all_str_edges is None:
        fallback_pool: set = set()
        for pool in candidates.values():
            fallback_pool |= set(pool)
    else:
        fallback_pool = set(all_str_edges)

    made_progress = True
    while made_progress:
        made_progress = False
        used_str = set(fixed.values())
        next_unresolved = []

        for e_ud in unresolved:
            exact_pool = candidates.get(e_ud, set())
            available = exact_pool - used_str

            if not available and allow_open_fallback:
                available = fallback_pool - used_str

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