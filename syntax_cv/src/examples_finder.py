"""
examples_finder.py — locate sentences that illustrate each of the five
categories raised by the supervisor on 15-Apr-26.

Categories (quoted from the supervisor's message):

  1. "Для предложения, где структуры изначально совпадают."
     — every non-punct UD edge has exact_same_dir in strict mode.

  2. "Для предложения, где структуры стали совпадать после первого
     подхода."
     — strict mode produces only {exact_same_dir, exact_mirrored}
     matches, i.e. every edge is resolved by CP elimination over
     identical-endpoint candidates, but at least one is mirrored.

  3. "Для предложения, где всем словам соответствует паттерн во втором
     подходе, хотя после первого подхода совпадения структур не
     получилось."
     — strict mode leaves ≥ 1 unresolved edge; extended mode (LCA
     candidates) fixes every edge.

  4. "Для предложения, где структуры стали совпадать после первого
     подхода, но обнаружить все паттерны вторым подходом не получилось
     (или просто убедиться, что такое действительно ни разу не
     встретилось при реализации алгоритма)."
     — strict mode resolves all edges, but extended mode (when run
     independently with the same inputs) fails to resolve some. If the
     extended candidates are a SUPERSET of the strict ones (which they
     are, by construction in `build_lca_candidates` plus automatic
     exact inclusion in strict), this category should be empty. We
     provide a verification routine that returns the (usually empty)
     list so the supervisor's "просто убедиться" is answered with data.

  5. "Для предложения, где ни один подход не сработал."
     — both strict and extended leave at least one edge unresolved.

All examples are returned as sent_ids with a diagnostic dict
per sentence (counts per status in each mode). The caller can feed
a sent_id into `tikz_dep.render_pair` for a picture.

Nothing in this module modifies input DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.edge_correspondence import (
    EdgeCorrespondence,
    SentID,
    build_edge_correspondence,
)
from src.lca_candidates import build_lca_candidates


# ---------------------------------------------------------------------------
# Category classification per sentence
# ---------------------------------------------------------------------------

@dataclass
class SentenceDiagnosis:
    sent_id: SentID
    strict_counts: dict[str, int]    # status → count
    extended_counts: dict[str, int]
    strict_total: int
    extended_total: int
    strict_meaningful_counts: dict[str, int]
    extended_meaningful_counts: dict[str, int]
    token_case_counts: dict[int, int] # per-token cases 1..5
    meaningful_tokens: int            # UD tokens: non-punct, including root
    category: int                     # 1..5


def _counts(corr: EdgeCorrespondence, sent_id: SentID) -> tuple[dict[str, int], int]:
    sc = corr.per_sentence.get(sent_id)
    if sc is None:
        return ({}, 0)
    by: dict[str, int] = {}
    for m in sc.matches.values():
        by[m.status] = by.get(m.status, 0) + 1
    return by, sum(by.values())


def _root_tokens(sk) -> set[int]:
    """
    Root token ids in a sentence skeleton.
    """
    dependents = {d for d in (sk.dep_of(e) for e in sk.edges) if d is not None}
    return {v for v in sk.token_ids if v not in dependents}


def _counts_meaningful(corr: EdgeCorrespondence, sent_id: SentID) -> tuple[dict[str, int], int]:
    """
    Status counts on meaningful tokens only (no punct, root included).
    """
    status_map = _meaningful_status_map(corr, sent_id)
    by: dict[str, int] = {
        "exact_same_dir": 0,
        "exact_mirrored": 0,
        "restructured": 0,
        "unresolved": 0,
    }
    for st in status_map.values():
        by[st] = by.get(st, 0) + 1
    return by, len(status_map)


def _meaningful_status_map(corr: EdgeCorrespondence, sent_id: SentID) -> dict[int, str]:
    """
    Per-token status on meaningful UD tokens (no punct, root included).
    """
    out: dict[int, str] = {}
    sc = corr.per_sentence.get(sent_id)
    ud_sk = corr.ud_skel.get(sent_id)
    if ud_sk is None:
        return out

    punct = corr.ud_punct_tokens.get(sent_id, set())
    if sc is not None:
        for e_ud, m in sc.matches.items():
            dep_ud = ud_sk.dep_of(e_ud)
            if dep_ud is None or dep_ud in punct:
                continue
            out[int(dep_ud)] = m.status

    # Add root token(s): include in meaningful stats as requested.
    ud_roots = _root_tokens(ud_sk)
    str_sk = corr.str_skel.get(sent_id)
    str_roots = _root_tokens(str_sk) if str_sk is not None else set()
    for v in ud_roots:
        if v in punct:
            continue
        out[int(v)] = "exact_same_dir" if v in str_roots else "unresolved"

    return out


def _token_case_counts(
    strict: EdgeCorrespondence,
    extended: EdgeCorrespondence,
    sent_id: SentID,
) -> tuple[dict[int, int], int]:
    """
    Per-token category counts (cases 1..5), meaningful tokens only.

    Case mapping is token-level and forms a partition:
      1: strict exact_same_dir
      2: strict resolved but not exact_same_dir, extended resolved
      3: strict unresolved, extended resolved
      4: strict resolved, extended unresolved
      5: strict unresolved, extended unresolved
    """
    s_map = _meaningful_status_map(strict, sent_id)
    e_map = _meaningful_status_map(extended, sent_id)
    token_ids = set(s_map) | set(e_map)
    by_case = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for tid in token_ids:
        s = s_map.get(tid, "unresolved")
        e = e_map.get(tid, "unresolved")

        s_resolved = s in ("exact_same_dir", "exact_mirrored", "restructured")
        e_resolved = e in ("exact_same_dir", "exact_mirrored", "restructured")

        if s == "exact_same_dir":
            by_case[1] += 1
        elif s_resolved and e_resolved:
            by_case[2] += 1
        elif (not s_resolved) and e_resolved:
            by_case[3] += 1
        elif s_resolved and (not e_resolved):
            by_case[4] += 1
        else:
            by_case[5] += 1

    return by_case, len(token_ids)


def _classify(
    strict_by: dict[str, int],
    strict_total: int,
    extended_by: dict[str, int],
    extended_total: int,
) -> int:
    """
    Return the category index 1..5 (or 0 if the sentence is empty).

    Priority order: 1 > 2 > 3 > 4 > 5. Category 4 is the supervisor's
    "should never happen" case; we report it if it ever does.
    """
    if strict_total == 0:
        return 0

    strict_unresolved = strict_by.get("unresolved", 0)
    strict_mirrored = strict_by.get("exact_mirrored", 0)
    strict_restr = strict_by.get("restructured", 0)
    ext_unresolved = extended_by.get("unresolved", 0)

    if strict_unresolved == 0 and strict_mirrored == 0 and strict_restr == 0:
        return 1  # full exact_same_dir

    if strict_unresolved == 0 and (strict_mirrored > 0 or strict_restr > 0):
        # structurally resolved by CP over exact candidates
        if ext_unresolved == 0:
            return 2
        # strict resolves everything but extended does not → category 4
        return 4

    # strict has at least one unresolved
    if ext_unresolved == 0:
        return 3

    return 5


def diagnose_sentences(
    corr_s,
    corr_e,
) -> dict[SentID, SentenceDiagnosis]:
    """
    Build both correspondences and classify every sentence.
    """
    strict = corr_s
    extended = corr_e

    all_sent_ids = set(strict.per_sentence) | set(extended.per_sentence)

    out: dict[SentID, SentenceDiagnosis] = {}
    for sid in all_sent_ids:
        s_by, s_tot = _counts(strict, sid)
        e_by, e_tot = _counts(extended, sid)
        s_mean_by, s_mean_tot = _counts_meaningful(strict, sid)
        e_mean_by, e_mean_tot = _counts_meaningful(extended, sid)
        token_cases, n_mean = _token_case_counts(strict, extended, sid)
        cat = _classify(s_by, s_tot, e_by, e_tot)
        if cat == 0:
            continue
        out[sid] = SentenceDiagnosis(
            sent_id=sid,
            strict_counts=s_by,
            extended_counts=e_by,
            strict_total=s_tot,
            extended_total=e_tot,
            strict_meaningful_counts=s_mean_by,
            extended_meaningful_counts=e_mean_by,
            token_case_counts=token_cases,
            meaningful_tokens=n_mean if n_mean else (min(s_mean_tot, e_mean_tot) if e_mean_tot else s_mean_tot),
            category=cat,
        )
    return out


# ---------------------------------------------------------------------------
# Category selection
# ---------------------------------------------------------------------------

def pick_examples(
    diagnoses: dict[SentID, SentenceDiagnosis],
    *,
    per_category: int = 1,
    min_tokens: int = 4,
    max_tokens: int = 12,
    ud_df: Optional[pd.DataFrame] = None,
) -> dict[int, list[SentID]]:
    """
    Pick representative sent_ids for each category 1..5.

    Preference order: sentences in [min_tokens, max_tokens] range, then
    by ascending total token count (shorter examples are easier to draw
    with TikZ). Pass `ud_df` to enable the token-count filter.
    """
    token_count: dict[SentID, int] = {}
    if ud_df is not None:
        # Accept both flat frames (sent_id/id columns) and MultiIndex
        # frames where sent_id/id live in the index.
        if "sent_id" in ud_df.columns:
            by_sent = ud_df.groupby("sent_id", sort=False)
            token_count = (
                by_sent["id"].count().to_dict()
                if "id" in ud_df.columns
                else by_sent.size().to_dict()
            )
        elif isinstance(ud_df.index, pd.MultiIndex) and "sent_id" in ud_df.index.names:
            token_count = ud_df.groupby(level="sent_id", sort=False).size().to_dict()
        elif ud_df.index.name == "sent_id":
            token_count = ud_df.groupby(level=0, sort=False).size().to_dict()
        else:
            raise KeyError("ud_df must contain sent_id either as a column or as an index level")

    buckets: dict[int, list[SentID]] = {c: [] for c in range(1, 6)}
    for sid, diag in diagnoses.items():
        n = token_count.get(sid, 1_000)
        if n < min_tokens or n > max_tokens:
            continue
        buckets[diag.category].append(sid)

    # Stable deterministic ordering: by token count, then by sent_id.
    for c in buckets:
        buckets[c].sort(key=lambda sid: (token_count.get(sid, 0), str(sid)))
        buckets[c] = buckets[c][:per_category]

    return buckets


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def category_summary(
    diagnoses: dict[SentID, SentenceDiagnosis],
) -> pd.DataFrame:
    """
    Aggregate category counts over all sentences. Answers the
    supervisor's implicit question: how often does each category occur?
    """
    rows = []
    total_sentences = len(diagnoses)
    total_meaningful_tokens = sum(d.meaningful_tokens for d in diagnoses.values())
    by_cat: dict[int, int] = {}
    by_cat_tokens: dict[int, int] = {}
    for d in diagnoses.values():
        by_cat[d.category] = by_cat.get(d.category, 0) + 1
        for cat in range(1, 6):
            by_cat_tokens[cat] = by_cat_tokens.get(cat, 0) + d.token_case_counts.get(cat, 0)
    for cat in range(1, 6):
        n_sent = by_cat.get(cat, 0)
        n_tok = by_cat_tokens.get(cat, 0)
        rows.append({
            "category": cat,
            "description": _CATEGORY_DESCRIPTIONS[cat],
            "sentences": n_sent,
            "sentences_pct": round(100 * n_sent / total_sentences, 2) if total_sentences else 0.0,
            "tokens": n_tok,
            "tokens_pct": round(100 * n_tok / total_meaningful_tokens, 2)
                          if total_meaningful_tokens else 0.0,
        })
    return pd.DataFrame(rows)


_CATEGORY_DESCRIPTIONS = {
    1: "Структуры изначально совпадают",
    2: "Первый метод справился",
    3: "Первый метод не справился, второй справился",
    4: "Первый метод справился, второй не справился",
    5: "Ни один метод не справился",
}
