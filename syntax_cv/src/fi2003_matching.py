"""
fi2003_matching.py — ordered-tree alignment matcher inspired by FI2003.

This module implements a practical dynamic-programming matcher based on the
ordered tree-alignment recurrences used in Jiang-Wang-Zhang / FI2003:

    DT(T1, T2) over tree pairs,
    D(F1, F2) over ordered forests,
    D'(F1, F2) for "both roots aligned with blanks" alternatives.

In addition, an optional k-relevance filter is applied to subtree pairs:
    |size(T1)-size(T2)| <= k and |left_leaves(T1)-left_leaves(T2)| <= k.
When k is not given, we use a small doubling schedule and fall back to the
unfiltered DP if needed.

The output is converted to the correspondence shape required by
edge_correspondence: a partial injective mapping e_ud -> e_str.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from src.edge_correspondence import Edge, SentenceSkeleton, TokenID


INF = 10**12


@dataclass(frozen=True)
class OrderedTree:
    roots: tuple[TokenID, ...]
    children: dict[TokenID, tuple[TokenID, ...]]
    subtree_size: dict[TokenID, int]
    left_leaves: dict[TokenID, int]


def _roots(sk: SentenceSkeleton) -> tuple[TokenID, ...]:
    deps = {d for d in (sk.dep_of(e) for e in sk.edges) if d is not None}
    return tuple(sorted(t for t in sk.token_ids if t not in deps))


def _build_ordered_tree(sk: SentenceSkeleton) -> OrderedTree:
    children_tmp: dict[TokenID, list[TokenID]] = {t: [] for t in sk.token_ids}
    for e in sk.edges:
        h = sk.head_of(e)
        d = sk.dep_of(e)
        if h is None or d is None:
            continue
        children_tmp.setdefault(h, []).append(d)
        children_tmp.setdefault(d, [])

    children: dict[TokenID, tuple[TokenID, ...]] = {
        t: tuple(sorted(ch)) for t, ch in children_tmp.items()
    }
    roots = _roots(sk)

    subtree_size: dict[TokenID, int] = {}

    def dfs_size(u: TokenID) -> int:
        s = 1
        for c in children.get(u, ()):
            s += dfs_size(c)
        subtree_size[u] = s
        return s

    for r in roots:
        dfs_size(r)

    left_leaves: dict[TokenID, int] = {}
    leaf_ix = 0

    def dfs_left(u: TokenID) -> None:
        nonlocal leaf_ix
        first = leaf_ix
        ch = children.get(u, ())
        if not ch:
            leaf_ix += 1
        else:
            for c in ch:
                dfs_left(c)
        left_leaves[u] = first

    for r in roots:
        dfs_left(r)

    return OrderedTree(
        roots=roots,
        children=children,
        subtree_size=subtree_size,
        left_leaves=left_leaves,
    )


class _FI2003Aligner:
    def __init__(
        self,
        t_str: OrderedTree,
        t_ud: OrderedTree,
        *,
        k_limit: Optional[int],
        mismatch_cost: int,
        gap_cost: int,
    ) -> None:
        self.t_str = t_str
        self.t_ud = t_ud
        self.k_limit = k_limit
        self.mismatch_cost = mismatch_cost
        self.gap_cost = gap_cost

    def _d_label(self, s: TokenID, u: TokenID) -> int:
        return 0 if s == u else self.mismatch_cost

    def _d_gap_del(self, _s: TokenID) -> int:
        return self.gap_cost

    def _d_gap_ins(self, _u: TokenID) -> int:
        return self.gap_cost

    def _relevant(self, s: TokenID, u: TokenID) -> bool:
        if self.k_limit is None:
            return True
        return (
            abs(self.t_str.subtree_size[s] - self.t_ud.subtree_size[u]) <= self.k_limit
            and abs(self.t_str.left_leaves[s] - self.t_ud.left_leaves[u]) <= self.k_limit
        )

    @lru_cache(maxsize=None)
    def dt(self, s: Optional[TokenID], u: Optional[TokenID]) -> int:
        if s is None and u is None:
            return 0
        if s is None:
            assert u is not None
            fu = self.t_ud.children.get(u, ())
            return self.d((), fu) + self._d_gap_ins(u)
        if u is None:
            fs = self.t_str.children.get(s, ())
            return self.d(fs, ()) + self._d_gap_del(s)

        if not self._relevant(s, u):
            return INF

        fs = self.t_str.children.get(s, ())
        fu = self.t_ud.children.get(u, ())

        opt_match = self.d(fs, fu) + self._d_label(s, u)

        opt_u_gap = INF
        if fs:
            base = self.dt(s, None)
            best = min(self.dt(t, u) - self.dt(t, None) for t in fs)
            opt_u_gap = base + best

        opt_s_gap = INF
        if fu:
            base = self.dt(None, u)
            best = min(self.dt(s, t) - self.dt(None, t) for t in fu)
            opt_s_gap = base + best

        return min(opt_match, opt_u_gap, opt_s_gap)

    @lru_cache(maxsize=None)
    def d(self, fs: tuple[TokenID, ...], fu: tuple[TokenID, ...]) -> int:
        if not fs and not fu:
            return 0
        if not fs:
            return sum(self.dt(None, t) for t in fu)
        if not fu:
            return sum(self.dt(t, None) for t in fs)

        fs0, ts = fs[:-1], fs[-1]
        fu0, tu = fu[:-1], fu[-1]

        opt_del = self.d(fs0, fu) + self.dt(ts, None)
        opt_ins = self.d(fs, fu0) + self.dt(None, tu)
        opt_match = self.d(fs0, fu0) + self.dt(ts, tu)
        opt_prime = self.d_prime(fs, fu)
        return min(opt_del, opt_ins, opt_match, opt_prime)

    @lru_cache(maxsize=None)
    def d_prime(self, fs: tuple[TokenID, ...], fu: tuple[TokenID, ...]) -> int:
        if not fs or not fu:
            return INF

        ts, tu = fs[-1], fu[-1]
        fs0, fu0 = fs[:-1], fu[:-1]
        fs_child = self.t_str.children.get(ts, ())
        fu_child = self.t_ud.children.get(tu, ())

        best_u_gap = INF
        # Corresponds to splitting fs into prefix + suffix, and replacing
        # fu's last root by its children (tu aligned to gap).
        for i in range(len(fs)):
            val = (
                self.d(fs[:i], fu0)
                + self.d(fs[i:], fu_child)
                + self._d_gap_ins(tu)
            )
            if val < best_u_gap:
                best_u_gap = val

        best_s_gap = INF
        # Symmetric case: ts aligned to gap.
        for j in range(len(fu)):
            val = (
                self.d(fs0, fu[:j])
                + self.d(fs_child, fu[j:])
                + self._d_gap_del(ts)
            )
            if val < best_s_gap:
                best_s_gap = val

        return min(best_u_gap, best_s_gap)

    def distance(self) -> int:
        return self.d(self.t_str.roots, self.t_ud.roots)

    @lru_cache(maxsize=None)
    def trace_dt(self, s: Optional[TokenID], u: Optional[TokenID]) -> frozenset[tuple[TokenID, TokenID]]:
        if s is None and u is None:
            return frozenset()
        if s is None:
            assert u is not None
            return self.trace_d((), self.t_ud.children.get(u, ()))
        if u is None:
            return self.trace_d(self.t_str.children.get(s, ()), ())

        target = self.dt(s, u)
        fs = self.t_str.children.get(s, ())
        fu = self.t_ud.children.get(u, ())

        candidates: list[tuple[str, int]] = []
        candidates.append(("match", self.d(fs, fu) + self._d_label(s, u)))
        if fs:
            candidates.append((
                "u_gap",
                self.dt(s, None) + min(self.dt(t, u) - self.dt(t, None) for t in fs),
            ))
        if fu:
            candidates.append((
                "s_gap",
                self.dt(None, u) + min(self.dt(s, t) - self.dt(None, t) for t in fu),
            ))

        kind = min(
            (k for k, val in candidates if val == target),
            key=lambda x: {"match": 0, "u_gap": 1, "s_gap": 2}[x],
        )

        if kind == "match":
            return frozenset({(s, u)}) | self.trace_d(fs, fu)
        if kind == "u_gap":
            t_best = min(
                fs,
                key=lambda t: (self.dt(t, u) - self.dt(t, None), abs(t - u), t),
            )
            return self.trace_dt(t_best, u)

        t_best = min(
            fu,
            key=lambda t: (self.dt(s, t) - self.dt(None, t), abs(s - t), t),
        )
        return self.trace_dt(s, t_best)

    @lru_cache(maxsize=None)
    def trace_d(self, fs: tuple[TokenID, ...], fu: tuple[TokenID, ...]) -> frozenset[tuple[TokenID, TokenID]]:
        if not fs and not fu:
            return frozenset()
        if not fs:
            out = frozenset()
            for t in fu:
                out |= self.trace_dt(None, t)
            return out
        if not fu:
            out = frozenset()
            for t in fs:
                out |= self.trace_dt(t, None)
            return out

        fs0, ts = fs[:-1], fs[-1]
        fu0, tu = fu[:-1], fu[-1]
        target = self.d(fs, fu)

        cands: list[tuple[str, int]] = [
            ("del", self.d(fs0, fu) + self.dt(ts, None)),
            ("ins", self.d(fs, fu0) + self.dt(None, tu)),
            ("match", self.d(fs0, fu0) + self.dt(ts, tu)),
            ("prime", self.d_prime(fs, fu)),
        ]
        kind = min(
            (k for k, val in cands if val == target),
            key=lambda x: {"match": 0, "prime": 1, "del": 2, "ins": 3}[x],
        )

        if kind == "del":
            return self.trace_d(fs0, fu) | self.trace_dt(ts, None)
        if kind == "ins":
            return self.trace_d(fs, fu0) | self.trace_dt(None, tu)
        if kind == "match":
            return self.trace_d(fs0, fu0) | self.trace_dt(ts, tu)
        return self.trace_d_prime(fs, fu)

    @lru_cache(maxsize=None)
    def trace_d_prime(
        self,
        fs: tuple[TokenID, ...],
        fu: tuple[TokenID, ...],
    ) -> frozenset[tuple[TokenID, TokenID]]:
        ts, tu = fs[-1], fu[-1]
        fs0, fu0 = fs[:-1], fu[:-1]
        fs_child = self.t_str.children.get(ts, ())
        fu_child = self.t_ud.children.get(tu, ())

        best_u_gap = INF
        best_i = 0
        for i in range(len(fs)):
            val = (
                self.d(fs[:i], fu0)
                + self.d(fs[i:], fu_child)
                + self._d_gap_ins(tu)
            )
            if val < best_u_gap:
                best_u_gap = val
                best_i = i

        best_s_gap = INF
        best_j = 0
        for j in range(len(fu)):
            val = (
                self.d(fs0, fu[:j])
                + self.d(fs_child, fu[j:])
                + self._d_gap_del(ts)
            )
            if val < best_s_gap:
                best_s_gap = val
                best_j = j

        if best_u_gap <= best_s_gap:
            return self.trace_d(fs[:best_i], fu0) | self.trace_d(fs[best_i:], fu_child)
        return self.trace_d(fs0, fu[:best_j]) | self.trace_d(fs_child, fu[best_j:])

    def aligned_pairs(self) -> set[tuple[TokenID, TokenID]]:
        return set(self.trace_d(self.t_str.roots, self.t_ud.roots))


def fi2003_match(
    str_sk: SentenceSkeleton,
    ud_sk: SentenceSkeleton,
    *,
    k_blank: Optional[int] = None,
    mismatch_cost: int = 3,
    gap_cost: int = 3,
) -> dict[Edge, Edge]:
    """
    Build e_ud -> e_str mapping using FI2003-style ordered-tree alignment.

    Parameters
    ----------
    k_blank : upper bound for k-relevance filtering. If None, tries a
              doubling schedule and then falls back to the unfiltered DP.
    mismatch_cost, gap_cost : label-distance parameters.
    """
    t_str = _build_ordered_tree(str_sk)
    t_ud = _build_ordered_tree(ud_sk)

    def _run(k: Optional[int]) -> tuple[int, set[tuple[TokenID, TokenID]]]:
        aligner = _FI2003Aligner(
            t_str=t_str,
            t_ud=t_ud,
            k_limit=k,
            mismatch_cost=mismatch_cost,
            gap_cost=gap_cost,
        )
        dist = aligner.distance()
        if dist >= INF:
            return dist, set()
        return dist, aligner.aligned_pairs()

    pairs: set[tuple[TokenID, TokenID]] = set()
    if k_blank is None:
        n = max(len(str_sk.token_ids), len(ud_sk.token_ids))
        k = max(1, abs(len(str_sk.token_ids) - len(ud_sk.token_ids)))
        while k <= 2 * max(1, n):
            dist, pairs = _run(k)
            if dist < INF:
                break
            k *= 2
        if not pairs:
            _, pairs = _run(None)
    else:
        _, pairs = _run(max(0, k_blank))

    ud_to_str: dict[TokenID, TokenID] = {}
    used_str_nodes: set[TokenID] = set()
    for s, u in sorted(pairs, key=lambda x: (x[1], x[0])):
        if u in ud_to_str or s in used_str_nodes:
            continue
        ud_to_str[u] = s
        used_str_nodes.add(s)

    incoming_str: dict[TokenID, Edge] = {}
    for e in str_sk.edges:
        d = str_sk.dep_of(e)
        if d is not None:
            incoming_str[d] = e

    used_str_edges: set[Edge] = set()
    fixed: dict[Edge, Edge] = {}

    ud_edges_sorted = sorted(
        ud_sk.edges.keys(),
        key=lambda e: (
            ud_sk.dep_of(e) if ud_sk.dep_of(e) is not None else -1,
            ud_sk.head_of(e) if ud_sk.head_of(e) is not None else -1,
        ),
    )

    for e_ud in ud_edges_sorted:
        d_ud = ud_sk.dep_of(e_ud)
        h_ud = ud_sk.head_of(e_ud)
        if d_ud is None:
            continue

        d_str = ud_to_str.get(d_ud)
        if d_str is None:
            continue

        chosen: Optional[Edge] = None
        if h_ud is not None:
            h_str = ud_to_str.get(h_ud)
            if h_str is not None:
                e_direct = frozenset({h_str, d_str})
                if e_direct in str_sk.edges and e_direct not in used_str_edges:
                    chosen = e_direct

        if chosen is None:
            e_in = incoming_str.get(d_str)
            if e_in is not None and e_in not in used_str_edges:
                chosen = e_in

        if chosen is not None:
            fixed[e_ud] = chosen
            used_str_edges.add(chosen)

    return fixed
