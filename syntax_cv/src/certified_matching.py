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


def build_certified_sentence_correspondence(
    sent_id,
    str_sk: SentenceSkeleton,
    ud_sk: SentenceSkeleton,
    *,
    lca_local_steps: Optional[list[dict[Edge, set[Edge]]]] = None,
    lca_full: Optional[dict[Edge, set[Edge]]] = None,
    fi_k: Optional[int] = None,
    use_fi2003: bool = False,
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
    support: dict[Edge, dict[Edge, dict[str, object]]] = {
        e_ud: {} for e_ud in ud_sk.edges
    }

    _add_exact_candidates(support, str_sk, ud_sk)
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

    fixed: dict[Edge, Edge] = {}

    fixed_phase1, still_unresolved = resolve_edge_matching(
        _candidate_pools(support),
        all_str_edges=None,
        allow_open_fallback=False,
        return_unresolved=True,
    )
    fixed.update(fixed_phase1)

    if still_unresolved and lca_full:
        for e_ud in still_unresolved:
            for e_str in lca_full.get(e_ud, ()):
                _add_candidate(support, e_ud, e_str, "lca_full")

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
        matches[e_ud] = _matched_edge(
            str_sk,
            ud_sk,
            e_ud,
            e_str,
            certification="strict",
            detail="singleton_cp",
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
        else:
            matches[e_ud] = EdgeMatch(
                e_ud=e_ud,
                e_str=None,
                status="candidate_gap",
                certification="candidate_gap",
                detail="no_candidate_after_completion",
                candidate_count=0,
                support_count=0,
            )

    component_sizes: list[tuple[int, int]] = []
    for comp_ud in _connected_components(residual_pools):
        component_sizes.append((len(comp_ud), len({s for u in comp_ud for s in residual_pools[u]})))
        comp_pools = {u: residual_pools[u] for u in comp_ud}

        _, struct_choices, struct_unmatched = _optimal_choice_sets(comp_pools)

        forced: dict[Edge, Edge] = {}
        structural_residual: dict[Edge, set[Edge]] = {}
        for e_ud in comp_ud:
            feasible = struct_choices[e_ud]
            if not feasible:
                matches[e_ud] = EdgeMatch(
                    e_ud=e_ud,
                    e_str=None,
                    status="candidate_gap",
                    certification="candidate_gap",
                    detail="not_in_structural_optimum",
                    candidate_count=0,
                    support_count=0,
                )
                continue

            if len(feasible) == 1 and not struct_unmatched[e_ud]:
                forced[e_ud] = next(iter(feasible))
            else:
                structural_residual[e_ud] = set(feasible)

        for e_ud, e_str in forced.items():
            meta = support.get(e_ud, {}).get(e_str, {})
            matches[e_ud] = _matched_edge(
                str_sk,
                ud_sk,
                e_ud,
                e_str,
                certification="strict",
                detail="forced_structural",
                candidate_count=1,
                support_count=len(meta.get("sources", ())),
            )

        if not structural_residual:
            continue

        forced_str = set(forced.values())
        heuristic_pools = {
            e_ud: {e_str for e_str in pool if e_str not in forced_str}
            for e_ud, pool in structural_residual.items()
        }

        # If structural feasibility depended only on edges that have just been
        # fixed globally, the residual pool collapses honestly.
        for e_ud, pool in list(heuristic_pools.items()):
            if pool:
                continue
            matches[e_ud] = EdgeMatch(
                e_ud=e_ud,
                e_str=None,
                status="candidate_gap",
                certification="candidate_gap",
                detail="exhausted_by_forced_edges",
                candidate_count=0,
                support_count=0,
            )
            heuristic_pools.pop(e_ud)

        if not heuristic_pools:
            continue

        weights = {
            e_ud: {
                e_str: _candidate_weight(str_sk, ud_sk, e_ud, e_str, support[e_ud][e_str])
                for e_str in pool
            }
            for e_ud, pool in heuristic_pools.items()
        }
        _, heur_choices, heur_unmatched = _optimal_choice_sets(heuristic_pools, weights=weights)

        for e_ud in heuristic_pools:
            feasible = heur_choices[e_ud]
            if len(feasible) == 1 and not heur_unmatched[e_ud]:
                e_str = next(iter(feasible))
                meta = support.get(e_ud, {}).get(e_str, {})
                matches[e_ud] = _matched_edge(
                    str_sk,
                    ud_sk,
                    e_ud,
                    e_str,
                    certification="heuristic",
                    detail="selected_by_secondary_objective",
                    candidate_count=1,
                    support_count=len(meta.get("sources", ())),
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
    }

    return SentenceCorrespondence(sent_id=sent_id, matches=matches, diagnostics=diagnostics)


def _add_exact_candidates(
    support: dict[Edge, dict[Edge, dict[str, object]]],
    str_sk: SentenceSkeleton,
    ud_sk: SentenceSkeleton,
) -> None:
    for e_ud in ud_sk.edges:
        if e_ud in str_sk.edges:
            _add_candidate(support, e_ud, e_ud, "exact", radius=1)


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
) -> tuple[tuple[int, int], dict[Edge, set[Edge]], dict[Edge, bool]]:
    """
    Return the best lexicographic objective and, for every UD edge, the set
    of STR candidates that appear in at least one optimal solution.

    Objective:
      1. maximize cardinality;
      2. maximize summed secondary weight.
    """
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

    return best_obj, choices, unmatched


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
    method_agreement = int("topdown" in sources and "fi2003" in sources)
    has_full_lca = int("lca_full" in sources)

    return (
        support_count * 1_000_000
        + method_agreement * 100_000
        + same_dep * 10_000
        + overlap * 1_000
        + locality_bonus * 10
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
