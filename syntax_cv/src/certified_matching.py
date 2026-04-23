"""
certified_matching.py — certified candidate-union alignment for
SynTagRus↔UD edge correspondence.

The goal of this module is not to "guess one best matching" immediately,
but to separate three layers of evidence:

  1. strict      — forced by structural constraints;
  2. heuristic   — uniquely selected only after a secondary scoring step;
  3. ambiguous   — still non-unique even after the heuristic layer.

The module consumes immutable sentence skeletons and returns a
SentenceCorrespondence compatible with the rest of the project.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from src.edge_correspondence import (
    Edge,
    EdgeMatch,
    SentenceCorrespondence,
    SentenceSkeleton,
    _build_head_bridge_candidates,
    _build_mirror_swap_candidates,
    _build_strict_plus_candidates,
)
from src.fi2003_matching import fi2003_match
from src.resolve import resolve_edge_matching
from src.topdown_matching import topdown_match

_EXACT_MAX_UD_EDGES = 16
_EXACT_MAX_STR_EDGES = 16
_EXACT_MAX_STATE_ESTIMATE = 1_000_000


def build_certified_sentence_correspondence(
    sent_id,
    str_sk: SentenceSkeleton,
    ud_sk: SentenceSkeleton,
    *,
    punct_tokens: Optional[set[int]] = None,
    lca_local_steps: Optional[list[dict[Edge, set[Edge]]]] = None,
    lca_full: Optional[dict[Edge, set[Edge]]] = None,
    fi_k: Optional[int] = None,
    use_fi2003: bool = False,
    use_punct_band: bool = True,
) -> SentenceCorrespondence:
    """
    Build certified per-sentence correspondence.

    Parameters
    ----------
    lca_local_steps
        Sentence-local LCA candidate pools with increasing radius
        (typically <=2, <=3, ...). Used as primary local evidence.
    lca_full
        Sentence-local unbounded LCA candidate pool used only on the
        residual set after the local pass. This is the completeness pass
        from the technical review.
    fi_k
        k-limit for FI2003-style auxiliary candidates.
    """
    punct_tokens = set(punct_tokens or set())
    # Backward compatibility: the punctuation-specific residual layer has
    # been removed, but the parameter is kept to avoid breaking old callers.
    _ = use_punct_band
    support: dict[Edge, dict[Edge, dict[str, object]]] = {
        e_ud: {} for e_ud in ud_sk.edges
    }

    _add_exact_same_dir_candidates(support, str_sk, ud_sk)
    _add_exact_mirrored_candidates(support, str_sk, ud_sk)

    fixed: dict[Edge, Edge] = {}

    fixed_exact, still_unresolved = resolve_edge_matching(
        _candidate_pools(support),
        all_str_edges=None,
        allow_open_fallback=False,
        return_unresolved=True,
    )
    fixed.update(fixed_exact)

    _add_candidate_family(support, _build_mirror_relative_candidates(str_sk, ud_sk), "mirror_relative")
    _add_candidate_family(support, _build_root_path_rotation_candidates(str_sk, ud_sk), "root_path")
    _add_candidate_family(support, _build_strict_plus_candidates(str_sk, ud_sk), "strict_plus")
    _add_candidate_family(support, _build_head_bridge_candidates(str_sk, ud_sk), "head_bridge")
    _add_candidate_family(support, _build_mirror_swap_candidates(str_sk, ud_sk), "mirror_swap")

    for step_idx, step in enumerate(lca_local_steps or [], start=2):
        _add_candidate_family(
            support,
            step,
            f"lca_step_{step_idx}",
            radius=step_idx,
        )

    _add_mapping_candidates(support, topdown_match(str_sk, ud_sk), "topdown")
    if use_fi2003:
        _add_mapping_candidates(support, fi2003_match(str_sk, ud_sk, k_blank=fi_k), "fi2003")

    if still_unresolved:
        _add_candidate_family(
            support,
            _build_order_residual_candidates(
                str_sk,
                ud_sk,
                punct_tokens=punct_tokens,
                unresolved_ud_edges=still_unresolved,
                unavailable_str_edges=set(fixed.values()),
            ),
            "order_residual",
        )

    fixed_phase1, still_unresolved = resolve_edge_matching(
        _candidate_pools(support, exclude_str=set(fixed.values()), only_edges=still_unresolved),
        all_str_edges=None,
        allow_open_fallback=False,
        return_unresolved=True,
    )
    fixed.update(fixed_phase1)

    if still_unresolved and lca_full:
        for e_ud in still_unresolved:
            for e_str in lca_full.get(e_ud, ()):
                _add_candidate(support, e_ud, e_str, "lca_full")

        _add_candidate_family(
            support,
            _build_order_residual_candidates(
                str_sk,
                ud_sk,
                punct_tokens=punct_tokens,
                unresolved_ud_edges=still_unresolved,
                unavailable_str_edges=set(fixed.values()),
            ),
            "order_residual",
        )

        fixed_phase2, still_unresolved = resolve_edge_matching(
            _candidate_pools(support, exclude_str=set(fixed.values()), only_edges=still_unresolved),
            all_str_edges=None,
            allow_open_fallback=False,
            return_unresolved=True,
        )
        fixed.update(fixed_phase2)

    matches: dict[Edge, EdgeMatch] = {}
    used_str = set(fixed.values())

    for e_ud, e_str in fixed.items():
        meta = support.get(e_ud, {}).get(e_str, {})
        certification = _certification_from_sources(meta)
        matches[e_ud] = _matched_edge(
            str_sk,
            ud_sk,
            e_ud,
            e_str,
            certification=certification,
            detail=(
                "singleton_cp"
                if certification == "strict"
                else "singleton_cp_heuristic_source"
            ),
            candidate_count=1,
            support_count=len(meta.get("sources", ())),
        )

    residual_pools: dict[Edge, set[Edge]] = {}
    for e_ud in ud_sk.edges:
        if e_ud in matches:
            continue
        pool = set(support.get(e_ud, {})) - used_str
        if pool:
            residual_pools[e_ud] = pool

    component_sizes: list[tuple[int, int]] = []
    structural_fallback_components = 0
    heuristic_fallback_components = 0
    heuristic_seed_pools: dict[Edge, set[Edge]] = {}
    forced_strict: dict[Edge, Edge] = {}
    for comp_ud in _connected_components(residual_pools):
        component_sizes.append((len(comp_ud), len({s for u in comp_ud for s in residual_pools[u]})))
        comp_pools = {u: residual_pools[u] for u in comp_ud}

        _, struct_choices, struct_unmatched, struct_exact = _optimal_choice_sets(comp_pools)
        if not struct_exact:
            structural_fallback_components += 1
            for e_ud in comp_ud:
                heuristic_seed_pools[e_ud] = set(comp_pools[e_ud])
            continue

        for e_ud in comp_ud:
            feasible = struct_choices[e_ud]
            if len(feasible) == 1 and not struct_unmatched[e_ud]:
                forced_strict[e_ud] = next(iter(feasible))
            else:
                heuristic_seed_pools[e_ud] = set(feasible)

    for e_ud, e_str in forced_strict.items():
        meta = support.get(e_ud, {}).get(e_str, {})
        certification = _certification_from_sources(meta)
        matches[e_ud] = _matched_edge(
            str_sk,
            ud_sk,
            e_ud,
            e_str,
            certification=certification,
            detail=(
                "forced_structural"
                if certification == "strict"
                else "forced_structural_heuristic_source"
            ),
            candidate_count=1,
            support_count=len(meta.get("sources", ())),
        )

    blocked_str = used_str | set(forced_strict.values())
    heuristic_edges = [e_ud for e_ud in ud_sk.edges if e_ud not in matches]

    soft_order = _build_order_residual_candidates(
        str_sk,
        ud_sk,
        punct_tokens=punct_tokens,
        unresolved_ud_edges=heuristic_edges,
        unavailable_str_edges=blocked_str,
    )
    _add_candidate_family(support, soft_order, "order_residual")

    heuristic_pools: dict[Edge, set[Edge]] = {}
    for e_ud in heuristic_edges:
        pool = {e_str for e_str in heuristic_seed_pools.get(e_ud, set()) if e_str not in blocked_str}
        pool |= {e_str for e_str in soft_order.get(e_ud, set()) if e_str not in blocked_str}
        if pool:
            heuristic_pools[e_ud] = pool
        else:
            matches[e_ud] = EdgeMatch(
                e_ud=e_ud,
                e_str=None,
                status="candidate_gap",
                certification="candidate_gap",
                detail="no_candidate_after_soft_completion",
                candidate_count=0,
                support_count=0,
            )

    for comp_ud in _connected_components(heuristic_pools):
        comp_pools = {u: heuristic_pools[u] for u in comp_ud}
        weights = {
            e_ud: {
                e_str: _candidate_weight(
                    str_sk,
                    ud_sk,
                    e_ud,
                    e_str,
                    support[e_ud][e_str],
                )
                for e_str in pool
            }
            for e_ud, pool in comp_pools.items()
        }
        _, heur_choices, heur_unmatched, heur_exact = _optimal_choice_sets(comp_pools, weights=weights)
        if not heur_exact:
            heuristic_fallback_components += 1

        for e_ud in comp_pools:
            feasible = heur_choices[e_ud]
            if len(feasible) == 1 and not heur_unmatched[e_ud]:
                e_str = next(iter(feasible))
                meta = support.get(e_ud, {}).get(e_str, {})
                sources = set(meta.get("sources", ()))
                certification = (
                    "strict"
                    if sources & {"exact_same_dir", "exact_mirrored"}
                    else "heuristic"
                )
                matches[e_ud] = _matched_edge(
                    str_sk,
                    ud_sk,
                    e_ud,
                    e_str,
                    certification=certification,
                    detail=(
                        "selected_by_secondary_objective"
                        if heur_exact
                        else "selected_by_weighted_fallback"
                    ),
                    candidate_count=1,
                    support_count=len(meta.get("sources", ())),
                )
            else:
                if not heur_exact:
                    matches[e_ud] = EdgeMatch(
                        e_ud=e_ud,
                        e_str=None,
                        status="candidate_gap",
                        certification="candidate_gap",
                        detail="unmatched_in_weighted_fallback",
                        candidate_count=0,
                        support_count=0,
                    )
                else:
                    matches[e_ud] = EdgeMatch(
                        e_ud=e_ud,
                        e_str=None,
                        status="ambiguous",
                        certification="ambiguous",
                        detail=(
                            "multiple_heuristic_optima"
                            if feasible or heur_unmatched[e_ud]
                            else "unclassified_residual"
                        ),
                        candidate_count=len(feasible),
                        support_count=_best_support_count(support, e_ud, feasible),
                    )

    diagnostics = {
        "ud_edges": len(ud_sk.edges),
        "str_edges": len(str_sk.edges),
        "strict": sum(1 for m in matches.values() if m.certification == "strict"),
        "heuristic": sum(1 for m in matches.values() if m.certification == "heuristic"),
        "ambiguous": sum(1 for m in matches.values() if m.certification == "ambiguous"),
        "candidate_gap": sum(1 for m in matches.values() if m.certification == "candidate_gap"),
        "components": len(component_sizes),
        "max_component_ud": max((u for u, _ in component_sizes), default=0),
        "max_component_str": max((s for _, s in component_sizes), default=0),
        "structural_fallback_components": structural_fallback_components,
        "heuristic_fallback_components": heuristic_fallback_components,
    }

    return SentenceCorrespondence(sent_id=sent_id, matches=matches, diagnostics=diagnostics)


def _add_exact_same_dir_candidates(
    support: dict[Edge, dict[Edge, dict[str, object]]],
    str_sk: SentenceSkeleton,
    ud_sk: SentenceSkeleton,
) -> None:
    for e_ud in ud_sk.edges:
        if e_ud not in str_sk.edges:
            continue
        if str_sk.head_of(e_ud) == ud_sk.head_of(e_ud):
            _add_candidate(support, e_ud, e_ud, "exact_same_dir", radius=1)


def _add_exact_mirrored_candidates(
    support: dict[Edge, dict[Edge, dict[str, object]]],
    str_sk: SentenceSkeleton,
    ud_sk: SentenceSkeleton,
) -> None:
    """
    Add shared-edge mirrored candidates as strict structural evidence.

    A same-endpoint mirrored edge means both corpora connect the same two
    tokens, but choose opposite heads. The edge is therefore comparable
    without any linguistic label assumptions; downstream rendering compares
    dependents, so this yields the intended crossed token pair.
    """
    for e_ud in ud_sk.edges:
        if e_ud not in str_sk.edges:
            continue
        if str_sk.head_of(e_ud) != ud_sk.head_of(e_ud):
            _add_candidate(support, e_ud, e_ud, "exact_mirrored", radius=1)


def _add_candidate_family(
    support: dict[Edge, dict[Edge, dict[str, object]]],
    family: dict[Edge, set[Edge]],
    source: str,
    *,
    radius: Optional[int] = None,
) -> None:
    for e_ud, pool in family.items():
        for e_str in pool:
            _add_candidate(support, e_ud, e_str, source, radius=radius)


def _add_mapping_candidates(
    support: dict[Edge, dict[Edge, dict[str, object]]],
    mapping: dict[Edge, Edge],
    source: str,
) -> None:
    for e_ud, e_str in mapping.items():
        _add_candidate(support, e_ud, e_str, source)


def _add_candidate(
    support: dict[Edge, dict[Edge, dict[str, object]]],
    e_ud: Edge,
    e_str: Edge,
    source: str,
    *,
    radius: Optional[int] = None,
) -> None:
    slot = support.setdefault(e_ud, {}).setdefault(
        e_str,
        {"sources": set(), "min_radius": None},
    )
    slot["sources"].add(source)
    if radius is not None:
        prev = slot.get("min_radius")
        slot["min_radius"] = radius if prev is None else min(prev, radius)


def _candidate_pools(
    support: dict[Edge, dict[Edge, dict[str, object]]],
    *,
    exclude_str: Optional[set[Edge]] = None,
    only_edges: Optional[list[Edge]] = None,
) -> dict[Edge, set[Edge]]:
    exclude = exclude_str or set()
    keys = only_edges if only_edges is not None else list(support.keys())
    return {
        e_ud: {e_str for e_str in support.get(e_ud, {}) if e_str not in exclude}
        for e_ud in keys
    }


def _build_root_path_rotation_candidates(
    str_sk: SentenceSkeleton,
    ud_sk: SentenceSkeleton,
) -> dict[Edge, set[Edge]]:
    """
    Candidates for root-shifted chain/star configurations.

    The rule is format-independent: it uses only roots and topology. If the
    STR root and the UD root are different, and the UD root lies below the
    STR root in STR, the STR path

        r_str -> x1 -> x2 -> ... -> r_ud

    can correspond to the UD star around r_ud:

        r_ud -> r_str, r_ud -> x1, r_ud -> x2, ...

    We therefore rotate the path by one step:

        UD {r_ud, path[i]}  ->  STR {path[i], path[i+1]}

    and also add fan candidates for shared children of path nodes:

        UD {r_ud, d}        ->  STR {path[i], d}

    The second part covers dependents that remain attached to the same local
    STR head after the root shift. This resolves cases like root-changing
    copular/coordination islands without consulting deprel names.
    """
    str_roots = _root_tokens(str_sk)
    ud_roots = _root_tokens(ud_sk)
    if len(str_roots) != 1 or len(ud_roots) != 1:
        return {}

    r_str = next(iter(str_roots))
    r_ud = next(iter(ud_roots))
    if r_str == r_ud:
        return {}

    h_str = _head_map(str_sk)
    path = _path_from_ancestor_to_descendant(h_str, ancestor=r_str, descendant=r_ud)
    if len(path) < 2:
        return {}

    out: dict[Edge, set[Edge]] = {}
    str_children = _children_by_head(str_sk)

    # Rotate the STR root-to-UD-root chain against the UD root-star.
    for i in range(len(path) - 1):
        e_ud = frozenset({r_ud, path[i]})
        e_str = frozenset({path[i], path[i + 1]})
        if e_ud in ud_sk.edges and e_str in str_sk.edges and e_ud != e_str:
            out.setdefault(e_ud, set()).add(e_str)

    # Add shared fan children under each non-terminal STR path node.
    path_set = set(path)
    for node in path[:-1]:
        for child in str_children.get(node, ()):
            if child in path_set:
                continue
            e_ud = frozenset({r_ud, child})
            e_str = frozenset({node, child})
            if e_ud in ud_sk.edges and e_str in str_sk.edges and e_ud != e_str:
                out.setdefault(e_ud, set()).add(e_str)

    return out


def _build_mirror_relative_candidates(
    str_sk: SentenceSkeleton,
    ud_sk: SentenceSkeleton,
) -> dict[Edge, set[Edge]]:
    """
    Candidates adjacent to exact mirrored edges.

    If the shared edge {a,b} is mirrored, then the neighbor edge entering
    the UD head side of that mirror is usually comparable to the neighbor
    edge entering the STR head side:

        UD:  p -> b -> a
        STR: p -> a -> b

    so UD {p,b} can be compared with STR {p,a}. This is a purely
    topological "mirror-relative" relation; it does not inspect deprel
    names such as cc/conj or case/obl.

    A second boundary rule handles the same pattern when a mirrored chain
    continues through an already shared bridge. If q is a STR child of the
    STR dependent side of the mirror, and p--q is an exact shared bridge,
    then the UD edge p->u, where u dominates the UD mirror head, may
    correspond to STR {mirror_dep, q}. This resolves root/coordination
    shifts where the continuation of the STR chain is the best counterpart
    for the top UD predicate above a mirrored island.
    """
    out: dict[Edge, set[Edge]] = {}
    h_ud = _head_map(ud_sk)
    h_str = _head_map(str_sk)
    children_str = _children_by_head(str_sk)
    shared_exact = {
        e
        for e in (set(str_sk.edges) & set(ud_sk.edges))
        if str_sk.head_of(e) == ud_sk.head_of(e)
    }

    for e in set(str_sk.edges) & set(ud_sk.edges):
        h_u = ud_sk.head_of(e)
        d_u = ud_sk.dep_of(e)
        h_s = str_sk.head_of(e)
        d_s = str_sk.dep_of(e)
        if h_u is None or d_u is None or h_s is None or d_s is None:
            continue
        if h_u == h_s:
            continue

        # For a same-endpoint mirrored edge, the UD head is the STR
        # dependent and the UD dependent is the STR head.
        if h_u != d_s or d_u != h_s:
            continue

        parent_ud = h_ud.get(h_u)
        if parent_ud is not None:
            e_ud = frozenset({parent_ud, h_u})
            e_str = frozenset({parent_ud, h_s})
            if e_ud in ud_sk.edges and e_str in str_sk.edges and e_ud != e_str:
                out.setdefault(e_ud, set()).add(e_str)

            # More general local mirror-neighbor: use the incoming STR edge
            # of the STR head side. This covers cases where UD presents a
            # root/star coordination but STR keeps a deeper coordination
            # chain, so the local STR parent is not the UD parent.
            parent_str = h_str.get(h_s)
            if parent_str is not None:
                e_str = frozenset({parent_str, h_s})
                if e_ud in ud_sk.edges and e_str in str_sk.edges and e_ud != e_str:
                    out.setdefault(e_ud, set()).add(e_str)

        # Boundary continuation through an exact shared bridge.
        ancestor = h_u
        while ancestor in h_ud:
            parent = h_ud[ancestor]
            e_ud = frozenset({parent, ancestor})
            for q in children_str.get(d_s, ()):
                e_bridge = frozenset({parent, q})
                e_str = frozenset({d_s, q})
                if (
                    e_ud in ud_sk.edges
                    and e_str in str_sk.edges
                    and e_bridge in shared_exact
                    and e_ud != e_str
                ):
                    out.setdefault(e_ud, set()).add(e_str)
            ancestor = parent

    return out


def _build_order_residual_candidates(
    str_sk: SentenceSkeleton,
    ud_sk: SentenceSkeleton,
    *,
    punct_tokens: set[int],
    unresolved_ud_edges: list[Edge],
    unavailable_str_edges: set[Edge],
) -> dict[Edge, set[Edge]]:
    """
    Soft candidates from the global residual dependent order.

    The generator works only with the order of non-punctuation dependents and
    does not split the sentence into punctuation-delimited bands.
    """
    ud_edges = sorted(
        [
            e_ud
            for e_ud in unresolved_ud_edges
            if (ud_sk.dep_of(e_ud) is not None and ud_sk.dep_of(e_ud) not in punct_tokens)
        ],
        key=lambda e: _dep_sort_key(ud_sk, e),
    )
    str_edges = sorted(
        [
            e_str
            for e_str in str_sk.edges
            if e_str not in unavailable_str_edges
            and (str_sk.dep_of(e_str) is not None and str_sk.dep_of(e_str) not in punct_tokens)
        ],
        key=lambda e: _dep_sort_key(str_sk, e),
    )

    out: dict[Edge, set[Edge]] = {}
    m = len(ud_edges)
    n = len(str_edges)
    if m == 0 or n == 0:
        return out

    for i, e_ud in enumerate(ud_edges):
        for j in _band_alignment_indices(i, m, n):
            if 0 <= j < n:
                out.setdefault(e_ud, set()).add(str_edges[j])

    return out


def _band_alignment_indices(i: int, m: int, n: int) -> set[int]:
    if n <= 0:
        return set()
    if m <= 1:
        center = 0 if n == 1 else (n - 1) / 2
    elif n <= 1:
        center = 0
    else:
        center = i * (n - 1) / (m - 1)

    rounded = round(center)
    out = {rounded}
    if m != n:
        out.add(int(center))
        out.add(int(center + 0.999999))
    out.add(rounded - 1)
    out.add(rounded + 1)
    return {j for j in out if 0 <= j < n}

def _dep_sort_key(sk: SentenceSkeleton, e: Edge) -> tuple[int, int, int]:
    dep = sk.dep_of(e)
    head = sk.head_of(e)
    return (
        dep if dep is not None else 10**9,
        head if head is not None else 10**9,
        *_edge_sort_key(e),
    )


def _head_map(sk: SentenceSkeleton) -> dict[int, int]:
    out: dict[int, int] = {}
    for e in sk.edges:
        dep = sk.dep_of(e)
        head = sk.head_of(e)
        if dep is not None and head is not None:
            out[int(dep)] = int(head)
    return out


def _root_tokens(sk: SentenceSkeleton) -> set[int]:
    dependents = {int(d) for d in (_head_map(sk).keys())}
    return {int(v) for v in sk.token_ids if int(v) not in dependents}


def _children_by_head(sk: SentenceSkeleton) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for e in sk.edges:
        head = sk.head_of(e)
        dep = sk.dep_of(e)
        if head is not None and dep is not None:
            out.setdefault(int(head), set()).add(int(dep))
    return out


def _path_from_ancestor_to_descendant(
    head_map: dict[int, int],
    *,
    ancestor: int,
    descendant: int,
) -> list[int]:
    reverse_path = [descendant]
    node = descendant
    while node in head_map:
        node = head_map[node]
        reverse_path.append(node)
        if node == ancestor:
            return list(reversed(reverse_path))
    return []


def _matched_edge(
    str_sk: SentenceSkeleton,
    ud_sk: SentenceSkeleton,
    e_ud: Edge,
    e_str: Edge,
    *,
    certification: str,
    detail: str,
    candidate_count: int,
    support_count: int,
) -> EdgeMatch:
    if e_ud == e_str:
        status = "exact_same_dir" if str_sk.head_of(e_str) == ud_sk.head_of(e_ud) else "exact_mirrored"
    else:
        status = "restructured"
    return EdgeMatch(
        e_ud=e_ud,
        e_str=e_str,
        status=status,
        certification=certification,
        detail=detail,
        candidate_count=candidate_count,
        support_count=support_count,
    )


def _connected_components(pools: dict[Edge, set[Edge]]) -> list[list[Edge]]:
    str_to_ud: dict[Edge, set[Edge]] = {}
    for e_ud, pool in pools.items():
        for e_str in pool:
            str_to_ud.setdefault(e_str, set()).add(e_ud)

    components: list[list[Edge]] = []
    seen_ud: set[Edge] = set()
    for start_ud in pools:
        if start_ud in seen_ud:
            continue
        stack_ud = [start_ud]
        comp_ud: set[Edge] = set()
        comp_str: set[Edge] = set()
        while stack_ud:
            e_ud = stack_ud.pop()
            if e_ud in comp_ud:
                continue
            comp_ud.add(e_ud)
            for e_str in pools.get(e_ud, ()):
                if e_str in comp_str:
                    continue
                comp_str.add(e_str)
                stack_ud.extend(str_to_ud.get(e_str, ()))
        seen_ud |= comp_ud
        components.append(sorted(comp_ud, key=_edge_sort_key))
    return components


def _optimal_choice_sets(
    pools: dict[Edge, set[Edge]],
    *,
    weights: Optional[dict[Edge, dict[Edge, int]]] = None,
) -> tuple[tuple[int, int], dict[Edge, set[Edge]], dict[Edge, bool], bool]:
    """
    Return the best lexicographic objective and, for every UD edge, the set
    of STR candidates that appear in at least one optimal solution.

    Objective:
      1. maximize cardinality;
      2. maximize summed secondary weight.

    For small components this function is exact and enumerates all optimal
    choices. For larger components it falls back to a polynomial assignment
    solver and returns one selected optimum only; in that case the last
    returned flag is False and the caller must not treat singleton results
    as structural certificates.
    """
    if not pools:
        return (0, 0), {}, {}, True

    if not _should_use_exact_solver(pools):
        return _optimal_choice_sets_fallback(pools, weights=weights)

    return _optimal_choice_sets_exact(pools, weights=weights)


def _optimal_choice_sets_exact(
    pools: dict[Edge, set[Edge]],
    *,
    weights: Optional[dict[Edge, dict[Edge, int]]] = None,
) -> tuple[tuple[int, int], dict[Edge, set[Edge]], dict[Edge, bool], bool]:
    ud_edges = sorted(pools, key=lambda e: (len(pools[e]), _edge_sort_key(e)))
    str_edges = sorted({e_str for pool in pools.values() for e_str in pool}, key=_edge_sort_key)
    str_ix = {e_str: i for i, e_str in enumerate(str_edges)}
    cand_ix = {
        e_ud: tuple(str_ix[e_str] for e_str in sorted(pools[e_ud], key=_edge_sort_key))
        for e_ud in ud_edges
    }
    weight_ix = {
        e_ud: {
            str_ix[e_str]: int(weights.get(e_ud, {}).get(e_str, 0) if weights else 0)
            for e_str in pools[e_ud]
        }
        for e_ud in ud_edges
    }
    max_gain = {
        e_ud: max(weight_ix[e_ud].values(), default=0)
        for e_ud in ud_edges
    }
    suffix_gain = [0] * (len(ud_edges) + 1)
    for i in range(len(ud_edges) - 1, -1, -1):
        suffix_gain[i] = suffix_gain[i + 1] + max_gain[ud_edges[i]]

    @lru_cache(maxsize=None)
    def solve(i: int, used_mask: int) -> tuple[int, int]:
        if i >= len(ud_edges):
            return (0, 0)

        e_ud = ud_edges[i]
        best = solve(i + 1, used_mask)
        for j in cand_ix[e_ud]:
            bit = 1 << j
            if used_mask & bit:
                continue
            size_rest, weight_rest = solve(i + 1, used_mask | bit)
            cand = (size_rest + 1, weight_rest + weight_ix[e_ud][j])
            if cand > best:
                best = cand
        return best

    best_obj = solve(0, 0)
    choices = {e_ud: set() for e_ud in ud_edges}
    unmatched = {e_ud: False for e_ud in ud_edges}

    reachable_masks = {0}
    for i, e_ud in enumerate(ud_edges):
        next_masks: set[int] = set()
        for used_mask in reachable_masks:
            best_here = solve(i, used_mask)
            if solve(i + 1, used_mask) == best_here:
                unmatched[e_ud] = True
                next_masks.add(used_mask)

            for j in cand_ix[e_ud]:
                bit = 1 << j
                if used_mask & bit:
                    continue
                size_rest, weight_rest = solve(i + 1, used_mask | bit)
                cand = (size_rest + 1, weight_rest + weight_ix[e_ud][j])
                if cand == best_here:
                    choices[e_ud].add(str_edges[j])
                    next_masks.add(used_mask | bit)
        reachable_masks = next_masks

    return best_obj, choices, unmatched, True


def _optimal_choice_sets_fallback(
    pools: dict[Edge, set[Edge]],
    *,
    weights: Optional[dict[Edge, dict[Edge, int]]] = None,
) -> tuple[tuple[int, int], dict[Edge, set[Edge]], dict[Edge, bool], bool]:
    ud_edges = sorted(pools, key=lambda e: (len(pools[e]), _edge_sort_key(e)))
    str_edges = sorted({e_str for pool in pools.values() for e_str in pool}, key=_edge_sort_key)

    choices = {e_ud: set() for e_ud in ud_edges}
    unmatched = {e_ud: False for e_ud in ud_edges}
    if not ud_edges:
        return (0, 0), choices, unmatched, False

    assignment, best_obj = _weighted_assignment_fallback(ud_edges, str_edges, pools, weights=weights)
    for e_ud, picked in assignment.items():
        if picked is None:
            unmatched[e_ud] = True
        else:
            choices[e_ud].add(picked)

    return best_obj, choices, unmatched, False


def _should_use_exact_solver(pools: dict[Edge, set[Edge]]) -> bool:
    ud_count = len(pools)
    str_count = len({e_str for pool in pools.values() for e_str in pool})
    if ud_count > _EXACT_MAX_UD_EDGES or str_count > _EXACT_MAX_STR_EDGES:
        return False
    state_estimate = (ud_count + 1) * (1 << str_count)
    return state_estimate <= _EXACT_MAX_STATE_ESTIMATE


def _weighted_assignment_fallback(
    ud_edges: list[Edge],
    str_edges: list[Edge],
    pools: dict[Edge, set[Edge]],
    *,
    weights: Optional[dict[Edge, dict[Edge, int]]] = None,
) -> tuple[dict[Edge, Optional[Edge]], tuple[int, int]]:
    """
    Memory-safe polynomial fallback for large components.

    We solve a rectangular assignment problem with one dummy column per UD
    edge. Real edges receive a huge cardinality bonus plus the secondary
    weight; dummy assignments receive zero. This implements the same
    lexicographic objective as the exact solver, but returns only one chosen
    optimum instead of the whole set of optimal alternatives.
    """
    if not ud_edges:
        return {}, (0, 0)

    real_ix = {e_str: j for j, e_str in enumerate(str_edges, start=1)}
    m = len(ud_edges)
    n_real = len(str_edges)
    n_cols = n_real + m
    inf = 10**18
    big = 10**12

    cost = [[0] * (n_cols + 1) for _ in range(m + 1)]
    for i, e_ud in enumerate(ud_edges, start=1):
        row_weights = weights.get(e_ud, {}) if weights else {}
        pool = set(pools.get(e_ud, set()))
        for j, e_str in enumerate(str_edges, start=1):
            if e_str not in pool:
                cost[i][j] = inf
                continue
            score = big + int(row_weights.get(e_str, 0))
            cost[i][j] = -score
        for j in range(n_real + 1, n_cols + 1):
            cost[i][j] = 0

    assignment_cols = _hungarian_min_cost(cost, m, n_cols)
    assignment: dict[Edge, Optional[Edge]] = {}
    matched = 0
    secondary = 0
    for i, e_ud in enumerate(ud_edges, start=1):
        col = assignment_cols[i]
        if 1 <= col <= n_real:
            e_str = str_edges[col - 1]
            assignment[e_ud] = e_str
            matched += 1
            secondary += int(weights.get(e_ud, {}).get(e_str, 0) if weights else 0)
        else:
            assignment[e_ud] = None
    return assignment, (matched, secondary)


def _hungarian_min_cost(cost: list[list[int]], n_rows: int, n_cols: int) -> list[int]:
    """
    Hungarian algorithm for rectangular matrices, minimizing total cost.

    `cost` is 1-indexed in both dimensions and must satisfy n_rows <= n_cols.
    Returns a 1-indexed list row -> chosen column.
    """
    u = [0] * (n_rows + 1)
    v = [0] * (n_cols + 1)
    p = [0] * (n_cols + 1)
    way = [0] * (n_cols + 1)

    for i in range(1, n_rows + 1):
        p[0] = i
        minv = [10**18] * (n_cols + 1)
        used = [False] * (n_cols + 1)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = 10**18
            j1 = 0
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = cost[i0][j] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [0] * (n_rows + 1)
    for j in range(1, n_cols + 1):
        if p[j] != 0:
            assignment[p[j]] = j
    return assignment


def _candidate_weight(
    str_sk: SentenceSkeleton,
    ud_sk: SentenceSkeleton,
    e_ud: Edge,
    e_str: Edge,
    meta: dict[str, object],
) -> int:
    sources = set(meta.get("sources", ()))
    support_count = len(sources)
    min_radius = meta.get("min_radius")
    if min_radius is None:
        locality_bonus = 0
    else:
        locality_bonus = max(0, 20 - int(min_radius))

    dep_ud = ud_sk.dep_of(e_ud)
    dep_str = str_sk.dep_of(e_str)
    overlap = len(e_ud & e_str)
    same_dep = int(dep_ud is not None and dep_str is not None and dep_ud == dep_str)
    exact_same = int(e_ud == e_str and str_sk.head_of(e_str) == ud_sk.head_of(e_ud))
    exact_mirrored = int(e_ud == e_str and str_sk.head_of(e_str) != ud_sk.head_of(e_ud))
    mirror_relative = int("mirror_relative" in sources)
    root_path = int("root_path" in sources)
    method_agreement = int("topdown" in sources and "fi2003" in sources)
    has_full_lca = int("lca_full" in sources)
    order_residual = int("order_residual" in sources)
    dep_distance = 0
    if dep_ud is not None and dep_str is not None:
        dep_distance = abs(dep_ud - dep_str)

    return (
        # User-requested heuristic priority:
        #   1) exact same-edge match
        #   2) mirrored same-edge match
        #   3) all other heuristic candidates
        # Exact same-dir edges are normally fixed before this function is
        # reached; the bonus is kept for completeness and for any future
        # callers that may score mixed pools directly.
        exact_same * 500_000_000
        + exact_mirrored * 100_000_000
        + mirror_relative * 20_000_000
        + root_path * 4_000_000
        + support_count * 1_000_000
        + same_dep * 500_000
        + order_residual * 10_000
        + method_agreement * 100_000
        + overlap * 1_000
        + locality_bonus * 10
        - dep_distance * 10
        - has_full_lca
    )


def _best_support_count(
    support: dict[Edge, dict[Edge, dict[str, object]]],
    e_ud: Edge,
    feasible: set[Edge],
) -> int:
    if not feasible:
        return 0
    return max(len(support.get(e_ud, {}).get(e_str, {}).get("sources", ())) for e_str in feasible)


def _edge_sort_key(e: Edge) -> tuple[int, int]:
    a, b = sorted(int(x) for x in e)
    return (a, b)


def _certification_from_sources(meta: dict[str, object]) -> str:
    sources = set(meta.get("sources", ()))
    if not sources:
        return "candidate_gap"
    if sources & {"exact_same_dir", "exact_mirrored"}:
        return "strict"
    if sources & {"mirror_relative", "root_path"}:
        return "heuristic"
    if sources <= {"order_residual"}:
        return "heuristic"
    return "strict"
