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
    category: int                     # 1..5


def _counts(corr: EdgeCorrespondence, sent_id: SentID) -> tuple[dict[str, int], int]:
    sc = corr.per_sentence.get(sent_id)
    if sc is None:
        return ({}, 0)
    by: dict[str, int] = {}
    for m in sc.matches.values():
        by[m.status] = by.get(m.status, 0) + 1
    return by, sum(by.values())


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
    str_df: pd.DataFrame,
    ud_df: pd.DataFrame,
    *,
    max_path_len: int = 3,
) -> dict[SentID, SentenceDiagnosis]:
    """
    Build both correspondences and classify every sentence.
    """
    strict = build_edge_correspondence(str_df, ud_df, mode="strict")
    extras = build_lca_candidates(str_df, ud_df, max_path_len=max_path_len)
    extended = build_edge_correspondence(
        str_df, ud_df, mode="extended", extra_candidates=extras,
    )

    all_sent_ids = set(strict.per_sentence) | set(extended.per_sentence)

    out: dict[SentID, SentenceDiagnosis] = {}
    for sid in all_sent_ids:
        s_by, s_tot = _counts(strict, sid)
        e_by, e_tot = _counts(extended, sid)
        cat = _classify(s_by, s_tot, e_by, e_tot)
        if cat == 0:
            continue
        out[sid] = SentenceDiagnosis(
            sent_id=sid,
            strict_counts=s_by,
            extended_counts=e_by,
            strict_total=s_tot,
            extended_total=e_tot,
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
    tot = len(diagnoses)
    by_cat: dict[int, int] = {}
    for d in diagnoses.values():
        by_cat[d.category] = by_cat.get(d.category, 0) + 1
    for cat in range(1, 6):
        n = by_cat.get(cat, 0)
        rows.append({
            "category": cat,
            "description": _CATEGORY_DESCRIPTIONS[cat],
            "sentences": n,
            "pct": round(100 * n / tot, 2) if tot else 0.0,
        })
    return pd.DataFrame(rows)


_CATEGORY_DESCRIPTIONS = {
    1: "Структуры изначально совпадают (все edges exact_same_dir)",
    2: "CP на exact-кандидатах разрешает всё (есть mirrored/restructured)",
    3: "Strict оставляет unresolved, extended (LCA) разрешает всё",
    4: "Strict разрешил всё, extended — нет (сигнал бага)",
    5: "Ни strict, ни extended не разрешают часть edges",
}
