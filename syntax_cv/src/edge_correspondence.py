"""
edge_correspondence.py — token-level alignment of SynTagRus and UD annotations
over IMMUTABLE dependency trees.

Philosophy
----------
The previous `align_markup.py` module modified the STR tree (swapping rows,
rewiring heads, producing "apply_swap_plan" / "apply_token_mapping" side
effects). That approach occasionally synthesised connections that do not
exist in either gold tree (e.g. an analytical "надо → бы" link introduced
only to satisfy swap bookkeeping), which is semantically wrong: SynTagRus
gold is a correct structure, UD gold is a correct structure, and token-level
comparison must not fabricate edges.

This module replaces that pipeline with a single artefact — an
`EdgeCorrespondence` — which is a partial injective map

                  φ : E_ud  ⇀  E_str

over the ORIGINAL edges of both sentences. Nothing in the input DataFrames
is mutated. All downstream analyses (label comparison, coverage statistics,
error mining) are expressed as lookups on φ.

Correctness condition (cf. method.md)
-------------------------------------
Let v ∈ V be a non-root token and let e_ud = {h_ud(v), v} be its incoming
UD edge. If φ(e_ud) = e_str and v is the dependent endpoint of e_str in
T_str, then comparing the two deprel labels is lingustically well-defined:
both labels describe the attachment of the same token v to one of its two
(possibly different) heads, and the heads are connected by a syntactic
relation that both formalisms recognise as the SAME relation.

Resolution strategies
---------------------
Two modes are supported, selected at construction time:

  • strict   — candidate set for e_ud is only the STR edge with identical
               endpoints (if it exists). Matches the original pair-based
               swap philosophy but without tree mutation. Unresolvable
               edges remain ⊥ and are excluded from comparison.

  • extended — candidate set additionally contains STR edges on the path
               from v to the cross-tree LCA. Handles "star ↔ chain"
               conjunction restructurings and similar cases where no
               single undirected-endpoint edge exists in STR.

In both modes the final fixation is performed by arc-consistency
constraint propagation (`resolve_edge_matching`, carried over from the
previous module): whenever an unresolved e_ud has exactly one surviving
candidate, it is fixed, which may in turn shrink the candidate pool of
other unresolved edges (the "method of exclusion" requested by the
supervisor).

The legacy fallback "all free STR edges" used in the previous code is
REMOVED — it was the source of the fabricated analytical link and is
philosophically incompatible with the "do not modify arrows" requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

import pandas as pd

from src.resolve import resolve_edge_matching


# ---------------------------------------------------------------------------
# Internal representations
# ---------------------------------------------------------------------------

Edge = frozenset          # {token_id_a, token_id_b}
TokenID = int
SentID = object           # string id in most corpora; kept generic


@dataclass(frozen=True)
class SentenceSkeleton:
    """
    Immutable per-sentence directed skeleton.

    edges      : undirected edge → head_id (the endpoint that governs).
    token_ids  : sorted list of all token ids in the sentence (excluding 0).
    """
    sent_id: SentID
    edges: dict[Edge, TokenID]
    token_ids: tuple[TokenID, ...]

    def head_of(self, e: Edge) -> Optional[TokenID]:
        return self.edges.get(e)

    def dep_of(self, e: Edge) -> Optional[TokenID]:
        h = self.edges.get(e)
        if h is None:
            return None
        a, b = tuple(e)
        return b if h == a else a


def _build_skeleton(df: pd.DataFrame) -> dict[SentID, SentenceSkeleton]:
    """
    Build per-sentence directed skeleton from a CoNLL-style DataFrame.

    Assumes columns: sent_id, id, head.
    head == 0 is treated as a virtual root; such edges are not included
    in the skeleton (they do not participate in cross-corpus comparison).
    """
    out: dict[SentID, SentenceSkeleton] = {}
    cols = df[["sent_id", "id", "head"]]
    for sent_id, grp in cols.groupby("sent_id", sort=False):
        ids = grp["id"].to_numpy(copy=False)
        heads = grp["head"].to_numpy(copy=False)
        edges: dict[Edge, TokenID] = {}
        for v, h in zip(ids, heads):
            h = int(h)
            if h == 0:
                continue
            v = int(v)
            edges[frozenset({h, v})] = h
        out[sent_id] = SentenceSkeleton(
            sent_id=sent_id,
            edges=edges,
            token_ids=tuple(sorted(int(x) for x in ids)),
        )
    return out


# ---------------------------------------------------------------------------
# EdgeCorrespondence — the single source of truth
# ---------------------------------------------------------------------------

MatchStatus = Literal[
    "exact_same_dir",   # same skeleton, same head direction
    "exact_mirrored",   # same skeleton, opposite head direction
    "restructured",     # φ(e_ud) ≠ e_ud; different endpoints, found via
                        #   extended candidates (e.g. LCA path)
    "unresolved",       # CP could not fix φ(e_ud) to a single candidate
    "ud_only_root",     # e_ud has head=0 (root edge); no STR counterpart
                        #   considered — roots are not comparable as "labels"
                        #   unless the STR root matches; handled separately.
]


@dataclass(frozen=True)
class EdgeMatch:
    e_ud: Edge
    e_str: Optional[Edge]
    status: MatchStatus


@dataclass
class SentenceCorrespondence:
    """
    Correspondence for a single sentence.

    matches     : dict e_ud → EdgeMatch (over ALL UD edges except root edge)
    diagnostics : compact per-sentence statistics (counts per status,
                  candidate pool sizes before CP, etc.)
    """
    sent_id: SentID
    matches: dict[Edge, EdgeMatch]
    diagnostics: dict = field(default_factory=dict)

    def resolved_count(self) -> int:
        return sum(
            1
            for m in self.matches.values()
            if m.status in ("exact_same_dir", "exact_mirrored", "restructured")
        )

    def unresolved_count(self) -> int:
        return sum(1 for m in self.matches.values() if m.status == "unresolved")

    def total_count(self) -> int:
        return len(self.matches)


@dataclass
class EdgeCorrespondence:
    """
    Whole-corpus correspondence. Keyed by sent_id.

    A corpus-level correspondence never rewrites the input DataFrames; it
    is attached to them from the outside and consumed by downstream
    analyses via the `.matches` / `.match_for_token` APIs.
    """
    mode: Literal["strict", "extended"]
    per_sentence: dict[SentID, SentenceCorrespondence]
    str_skel: dict[SentID, SentenceSkeleton]
    ud_skel: dict[SentID, SentenceSkeleton]
    ud_punct_tokens: dict[SentID, set[TokenID]] = field(default_factory=dict)

    # ---- public lookup API --------------------------------------------------

    def match_for_token(self, sent_id: SentID, v: TokenID) -> Optional[EdgeMatch]:
        """
        Return the EdgeMatch for the UD incoming edge of token v.

        Returns None when v has head=0 in UD (root token) or when v is not
        present in the sentence.
        """
        ud_sk = self.ud_skel.get(sent_id)
        if ud_sk is None:
            return None
        # find incoming UD edge
        for e_ud, head in ud_sk.edges.items():
            if head != v and v in e_ud and (e_ud - {v}).pop() == head:
                # v is the dependent endpoint in e_ud
                match = self.per_sentence[sent_id].matches.get(e_ud)
                return match
        return None

    def counterpart_str_token(
        self,
        sent_id: SentID,
        v: TokenID,
    ) -> Optional[TokenID]:
        """
        Given a UD token v, return the STR token whose incoming label is
        comparable with l_ud(v), or None when no such token can be chosen.

        Semantics per status:
          exact_same_dir : v          (dep remains the same token)
          exact_mirrored : h_ud(v)    (dep and head swap their roles)
          restructured   : dep endpoint of φ(e_ud) in T_str
          unresolved     : None
        """
        m = self.match_for_token(sent_id, v)
        if m is None or m.e_str is None:
            return None
        str_sk = self.str_skel[sent_id]
        return str_sk.dep_of(m.e_str)

    # ---- aggregate statistics ----------------------------------------------

    def coverage(
        self,
        *,
        include_punct: bool = False,
        include_root: bool = True,
        token_mask: Optional[dict[SentID, set[TokenID]]] = None,
    ) -> dict:
        """
        Aggregate coverage by match status.

        The unit is a UD token selected by filters below.

        Parameters
        ----------
        include_punct : include punctuation tokens in the denominator.
                        Defaults to False to match the legacy notebook
                        metric ("осмысленные токены").
        include_root  : include UD root token(s) in the denominator.
                        Defaults to True to match the legacy notebook
                        metric used in syntax_cv_old.
        token_mask    : optional per-sentence set of UD token ids to
                        include in the denominator/numerator.
        """
        counts: dict[str, int] = {
            "exact_same_dir": 0,
            "exact_mirrored": 0,
            "restructured": 0,
            "unresolved": 0,
        }
        total = 0

        for sent_id, sc in self.per_sentence.items():
            ud_sk = self.ud_skel[sent_id]
            allowed: Optional[set[TokenID]] = None
            if token_mask is not None:
                allowed = set(token_mask.get(sent_id, set()))
            else:
                allowed = set(ud_sk.token_ids)
            if not include_punct:
                allowed -= self.ud_punct_tokens.get(sent_id, set())

            for e_ud, m in sc.matches.items():
                dep_ud = ud_sk.dep_of(e_ud)
                if dep_ud is None:
                    continue
                if dep_ud not in allowed:
                    continue
                total += 1
                counts[m.status] = counts.get(m.status, 0) + 1

            if include_root:
                ud_roots = _root_tokens(ud_sk)
                str_roots = _root_tokens(self.str_skel[sent_id]) if sent_id in self.str_skel else set()
                for v in ud_roots:
                    if v not in allowed:
                        continue
                    total += 1
                    if v in str_roots:
                        counts["exact_same_dir"] += 1
                    else:
                        counts["unresolved"] += 1

        resolved = counts["exact_same_dir"] + counts["exact_mirrored"] + counts["restructured"]
        comparable_heads_match = counts["exact_same_dir"]
        return {
            "mode": self.mode,
            "total_ud_edges": total,
            "by_status": dict(counts),
            "resolved": resolved,
            "resolved_pct": (100 * resolved / total) if total else 0.0,
            "baseline_heads_match": comparable_heads_match,
            "baseline_pct": (100 * comparable_heads_match / total) if total else 0.0,
        }


def _root_tokens(sk: SentenceSkeleton) -> set[TokenID]:
    """
    Root token ids in a sentence skeleton.

    Tokens that never appear as dependent endpoint in non-root edges
    are roots (normally exactly one in a tree).
    """
    dependents = {d for d in (sk.dep_of(e) for e in sk.edges) if d is not None}
    return {v for v in sk.token_ids if v not in dependents}


def _collect_ud_punct_tokens(ud_df: pd.DataFrame) -> dict[SentID, set[TokenID]]:
    """
    Build sent_id -> token ids marked as punctuation in UD.
    """
    out: dict[SentID, set[TokenID]] = {}
    if "sent_id" not in ud_df.columns or "id" not in ud_df.columns:
        return out

    has_deprel = "deprel" in ud_df.columns
    has_upos = "upos" in ud_df.columns
    if not has_deprel and not has_upos:
        return out

    cols = ["sent_id", "id"]
    if has_deprel:
        cols.append("deprel")
    if has_upos:
        cols.append("upos")

    for sent_id, grp in ud_df[cols].groupby("sent_id", sort=False):
        is_punct = pd.Series(False, index=grp.index)
        if has_deprel:
            is_punct |= grp["deprel"].astype(str).str.lower() == "punct"
        if has_upos:
            is_punct |= grp["upos"].astype(str).str.upper() == "PUNCT"
        if is_punct.any():
            out[sent_id] = set(int(x) for x in grp.loc[is_punct, "id"].tolist())
    return out


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def build_edge_correspondence(
    str_df: pd.DataFrame,
    ud_df: pd.DataFrame,
    *,
    mode: Literal["strict", "extended"] = "strict",
    extra_candidates: Optional[dict[SentID, dict[Edge, Iterable[Edge]]]] = None,
) -> EdgeCorrespondence:
    """
    Build an EdgeCorrespondence from two CoNLL-style DataFrames.

    Neither `str_df` nor `ud_df` is modified. Both must share `sent_id`
    and `id` keys. Columns actually used: sent_id, id, head.

    Parameters
    ----------
    mode             : 'strict'    — only exact-endpoint candidates.
                       'extended'  — exact + caller-provided extras
                                     (e.g. LCA-derived candidates, see
                                     `lca_candidates.build_lca_candidates`).
    extra_candidates : nested dict sent_id -> { e_ud -> set(e_str) }
                       merged with exact candidates before CP. In
                       strict mode this argument is IGNORED by design
                       (to keep strict provably well-defined).

    Returns
    -------
    EdgeCorrespondence with per-sentence SentenceCorrespondence objects.
    """
    str_skel = _build_skeleton(str_df)
    ud_skel = _build_skeleton(ud_df)

    per_sent: dict[SentID, SentenceCorrespondence] = {}

    for sent_id, ud_sk in ud_skel.items():
        str_sk = str_skel.get(sent_id)
        if str_sk is None:
            # Whole STR sentence missing — mark all UD edges as unresolved.
            matches = {
                e_ud: EdgeMatch(e_ud=e_ud, e_str=None, status="unresolved")
                for e_ud in ud_sk.edges
            }
            per_sent[sent_id] = SentenceCorrespondence(sent_id=sent_id, matches=matches)
            continue

        all_str_edges = set(str_sk.edges.keys())

        # Build candidate pools
        candidates: dict[Edge, set[Edge]] = {}
        for e_ud in ud_sk.edges:
            pool: set[Edge] = set()
            if e_ud in str_sk.edges:
                pool.add(e_ud)
            if mode == "extended" and extra_candidates:
                extras = extra_candidates.get(sent_id, {}).get(e_ud)
                if extras:
                    pool |= set(extras)
                    # Only keep candidates that actually exist in STR
                    pool &= all_str_edges | {e_ud}
                    pool &= all_str_edges  # keep only real STR edges
            candidates[e_ud] = pool

        # Run CP WITHOUT the "all free STR edges" fallback — in the immutable
        # setting such a fallback is not a principled match and would
        # re-introduce the class of errors that prompted this refactor.
        fixed, unresolved = resolve_edge_matching(
            candidates,
            all_str_edges=None,          # disabled
            allow_open_fallback=False,
            return_unresolved=True,
        )

        matches: dict[Edge, EdgeMatch] = {}
        for e_ud in ud_sk.edges:
            if e_ud in fixed:
                e_str = fixed[e_ud]
                if e_str == e_ud:
                    head_ud = ud_sk.edges[e_ud]
                    head_str = str_sk.edges[e_str]
                    status: MatchStatus = (
                        "exact_same_dir" if head_str == head_ud else "exact_mirrored"
                    )
                else:
                    status = "restructured"
                matches[e_ud] = EdgeMatch(e_ud=e_ud, e_str=e_str, status=status)
            else:
                matches[e_ud] = EdgeMatch(e_ud=e_ud, e_str=None, status="unresolved")

        diag = {
            "ud_edges": len(ud_sk.edges),
            "str_edges": len(str_sk.edges),
            "fixed": len(fixed),
            "unresolved": len(unresolved),
        }
        per_sent[sent_id] = SentenceCorrespondence(
            sent_id=sent_id, matches=matches, diagnostics=diag
        )

    # Sentences present in STR but absent in UD are simply ignored; the
    # correspondence object only guarantees coverage over UD-side edges.
    return EdgeCorrespondence(
        mode=mode,
        per_sentence=per_sent,
        str_skel=str_skel,
        ud_skel=ud_skel,
        ud_punct_tokens=_collect_ud_punct_tokens(ud_df),
    )


# ---------------------------------------------------------------------------
# Backward compatibility: expose skeleton accessor for downstream modules
# ---------------------------------------------------------------------------

def build_directed_skeleton_by_sent(df: pd.DataFrame) -> dict[SentID, dict[Edge, TokenID]]:
    """
    Return the raw skeleton dict used internally.

    Kept for tooling that prefers working with the plain dict instead of the
    SentenceSkeleton dataclass.
    """
    return {sid: sk.edges for sid, sk in _build_skeleton(df).items()}
