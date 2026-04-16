"""
tikz_dep.py — generate tikz-dependency snippets for side-by-side
visualisation of SynTagRus and UD trees.

Supervisor's specification (15-Apr-26, 11:36):

    "Можно было сохранить порядок и линиями соединять пары токенов
     между верхней и нижней картинкой. Так лучше всего."

Layout:
    - Upper `\\begin{deptext}` = SynTagRus tokens in their ORIGINAL order.
    - Lower `\\begin{deptext}` = UD tokens in their ORIGINAL order.
      (Same surface order in both corpora — differences are structural,
      not positional.)
    - Above the upper row: STR arcs with SynTagRus labels.
    - Below the lower row: UD arcs with UD labels.
    - Dashed lines connecting STR dependent to UD dependent for every
      comparable edge in the EdgeCorrespondence. Colour-coded per
      match status.

No arrows are redrawn, repointed, or removed. Both trees are shown
exactly as they appear in the gold data. Structural disagreements
manifest as crossed or missing dashed connectors, not as edited arcs.

Output
------
A string containing a self-contained tikz-dependency picture. The caller
is expected to wrap it in a tikzpicture environment and a LaTeX document
with the `tikz-dependency` package loaded. We also provide a
`render_document` wrapper that does this automatically.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from src.edge_correspondence import (
    EdgeCorrespondence,
    Edge,
    SentID,
    TokenID,
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

_STATUS_STYLE = {
    "exact_same_dir": "solid, thick, gray!70",
    "exact_mirrored": "dashed, thick, orange!80!black",
    "restructured":   "dashed, thick, blue!60!black",
    "unresolved":     "dotted, thick, red!60!black",
}


def _guarded_draw_line(style: str, a: str, b: str) -> str:
    return (
        f"  \\ifcsname pgf@sh@ns@{a}\\endcsname"
        f"\\ifcsname pgf@sh@ns@{b}\\endcsname"
        f"\\draw[{style}] ({a}) -- ({b});"
        f"\\fi\\fi"
    )


def _guarded_draw_circle(style: str, a: str, radius: str = "2mm") -> str:
    return (
        f"  \\ifcsname pgf@sh@ns@{a}\\endcsname"
        f"\\draw[{style}] ({a}) circle ({radius});"
        f"\\fi"
    )


def _escape_latex(s: object) -> str:
    if s is None:
        return ""
    text = str(s)
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    )
    for a, b in replacements:
        text = text.replace(a, b)
    return text


def _sentence_tokens(df: pd.DataFrame, sent_id: SentID) -> pd.DataFrame:
    sub = df[df["sent_id"] == sent_id].copy()
    sub = sub.sort_values("id").reset_index(drop=True)
    return sub


def _as_render_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize input frame for rendering.

    Accepts both:
    - flat CoNLL-style frames with sent_id/id/head/deprel columns;
    - CV pair frames where sent_id/id are in the index and gold columns
      are named head_g/deprel_g.
    """
    out = df
    if "sent_id" not in out.columns or "id" not in out.columns:
        if isinstance(out.index, pd.MultiIndex) and {"sent_id", "id"}.issubset(set(out.index.names)):
            out = out.reset_index()
        else:
            raise KeyError("DataFrame must contain sent_id and id as columns or index levels")

    if "head" not in out.columns:
        if "head_g" in out.columns:
            out = out.assign(head=out["head_g"])
        elif "head_p" in out.columns:
            out = out.assign(head=out["head_p"])
        else:
            raise KeyError("DataFrame must contain head or head_g/head_p")

    if "deprel" not in out.columns:
        if "deprel_g" in out.columns:
            out = out.assign(deprel=out["deprel_g"])
        elif "deprel_p" in out.columns:
            out = out.assign(deprel=out["deprel_p"])
        else:
            raise KeyError("DataFrame must contain deprel or deprel_g/deprel_p")

    return out


def _position_map(tokens: pd.DataFrame) -> dict[TokenID, int]:
    """Map token id → 1-based column index in the `\\deptext` row."""
    return {int(r["id"]): i + 1 for i, r in tokens.iterrows()}


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

def render_pair(
    str_df: pd.DataFrame,
    ud_df: pd.DataFrame,
    sent_id: SentID,
    corr: EdgeCorrespondence,
    *,
    show_unresolved: bool = True,
    draw_root_link: bool = True,
    root_link_style: str = "densely dashed, thick, gray!70",
    deprel_col: str = "deprel",
    form_col: str = "form",
) -> str:
    """
    Render one side-by-side STR/UD tikz-dependency picture with
    correspondence lines.

    The picture uses two `\\begin{dependency}` blocks stacked vertically
    because tikz-dependency does not support two rows in a single
    `\\deptext`. The blocks are linked by TikZ `\\draw` commands that
    reference named nodes produced with `\\deptext[column sep=...][...]`.

    Parameters
    ----------
    str_df, ud_df : input CoNLL-style DataFrames (NOT modified).
    sent_id       : target sentence identifier.
    corr          : EdgeCorrespondence built from the same DataFrames.
    show_unresolved
        If True, draw unresolved correspondences as red dotted lines
        between UD-dep and the best-guess STR token (none — just render
        the UD edge as decorative, with no STR counterpart). If False,
        skip them entirely.
    draw_root_link
        If True, draw an additional connector between root tokens.
        This connector is decorative: roots are not part of the
        edge-based correspondence map.
    root_link_style
        TikZ style used for the decorative root connector.
    deprel_col, form_col : column names for labels and surface forms.
    """
    str_df = _as_render_view(str_df)
    ud_df = _as_render_view(ud_df)

    str_toks = _sentence_tokens(str_df, sent_id)
    ud_toks = _sentence_tokens(ud_df, sent_id)
    if str_toks.empty or ud_toks.empty:
        raise ValueError(f"sent_id={sent_id!r} not present in both DataFrames")

    str_pos = _position_map(str_toks)
    ud_pos = _position_map(ud_toks)

    str_forms = [_escape_latex(f) for f in str_toks[form_col]]
    ud_forms = [_escape_latex(f) for f in ud_toks[form_col]]

    # --- STR dependency block (top) -----------------------------------------

    lines: list[str] = []
    lines.append(r"% ---- SynTagRus tree (top) ----")
    lines.append(r"\begin{minipage}{\linewidth}")
    lines.append(r"\centering")
    lines.append(r"\begin{dependency}[remember picture]")
    lines.append(r"  \begin{deptext}[column sep=0.6cm]")
    lines.append("    " + " \\& ".join(str_forms) + r" \\")
    lines.append(r"  \end{deptext}")

    skipped_str_edges = 0
    for _, row in str_toks.iterrows():
        head = int(row["head"])
        tid = int(row["id"])
        if head == 0:
            lines.append(f"  \\deproot{{{str_pos[tid]}}}{{{_escape_latex(row[deprel_col])}}}")
        else:
            if head not in str_pos:
                # Happens when caller passes token-filtered tables where
                # some heads are missing from the sentence slice.
                skipped_str_edges += 1
                continue
            lines.append(
                f"  \\depedge{{{str_pos[head]}}}{{{str_pos[tid]}}}"
                f"{{{_escape_latex(row[deprel_col])}}}"
            )
    if skipped_str_edges:
        lines.append(f"% skipped STR edges with missing heads: {skipped_str_edges}")
    for col in sorted(set(str_pos.values())):
        lines.append(f"  \\coordinate (str-w-{col}) at (\\wordref{{1}}{{{col}}});")
    lines.append(r"\end{dependency}")
    lines.append(r"\end{minipage}")
    lines.append(r"\par")
    lines.append(r"\vspace{0.6em}")
    lines.append("")

    # --- UD dependency block (bottom) ---------------------------------------

    lines.append(r"% ---- UD tree (bottom) ----")
    lines.append(r"\begin{minipage}{\linewidth}")
    lines.append(r"\centering")
    lines.append(r"\begin{dependency}[remember picture]")
    lines.append(r"  \begin{deptext}[column sep=0.6cm]")
    lines.append("    " + " \\& ".join(ud_forms) + r" \\")
    lines.append(r"  \end{deptext}")

    skipped_ud_edges = 0
    for _, row in ud_toks.iterrows():
        head = int(row["head"])
        tid = int(row["id"])
        if head == 0:
            lines.append(f"  \\deproot[edge below]{{{ud_pos[tid]}}}{{{_escape_latex(row[deprel_col])}}}")
        else:
            if head not in ud_pos:
                skipped_ud_edges += 1
                continue
            lines.append(
                f"  \\depedge[edge below]{{{ud_pos[head]}}}{{{ud_pos[tid]}}}"
                f"{{{_escape_latex(row[deprel_col])}}}"
            )
    if skipped_ud_edges:
        lines.append(f"% skipped UD edges with missing heads: {skipped_ud_edges}")
    for col in sorted(set(ud_pos.values())):
        lines.append(f"  \\coordinate (ud-w-{col}) at (\\wordref{{1}}{{{col}}});")
    lines.append(r"\end{dependency}")
    lines.append(r"\end{minipage}")
    lines.append(r"\par")
    lines.append(r"\vspace{0.4em}")
    lines.append("")

    # --- correspondence lines -----------------------------------------------
    # We rely on tikz-dependency's automatic node naming:
    #   \wordref{row}{col}  →  word at (row=1, col)
    # The picture is split into two separate `dependency` environments;
    # to connect nodes across environments we must emit a surrounding
    # `tikzpicture` with overlay+remember picture. See render_document.

    lines.append(r"% ---- token correspondence (dashed) ----")
    lines.append(r"\begin{tikzpicture}[overlay, remember picture]")
    sc = corr.per_sentence.get(sent_id)
    if sc is not None:
        ud_sk = corr.ud_skel[sent_id]
        str_sk = corr.str_skel[sent_id]
        for e_ud, m in sc.matches.items():
            dep_ud = ud_sk.dep_of(e_ud)
            if dep_ud is None:
                continue
            if dep_ud not in ud_pos:
                continue
            if m.status == "unresolved" and not show_unresolved:
                continue
            style = _STATUS_STYLE[m.status]
            if m.e_str is None:
                # Decorative marker on UD token only
                lines.append(_guarded_draw_circle(style, f"ud-w-{ud_pos[dep_ud]}"))
                continue
            dep_str = str_sk.dep_of(m.e_str)
            if dep_str is None:
                continue
            if dep_str not in str_pos:
                continue
            lines.append(
                _guarded_draw_line(
                    style,
                    f"str-w-{str_pos[dep_str]}",
                    f"ud-w-{ud_pos[dep_ud]}",
                )
            )

    if draw_root_link:
        str_roots = [int(r["id"]) for _, r in str_toks.iterrows() if int(r["head"]) == 0]
        ud_roots = [int(r["id"]) for _, r in ud_toks.iterrows() if int(r["head"]) == 0]
        shared = sorted(set(str_roots) & set(ud_roots))
        if shared:
            for rid in shared:
                if rid in str_pos and rid in ud_pos:
                    lines.append(
                        _guarded_draw_line(
                            root_link_style,
                            f"str-w-{str_pos[rid]}",
                            f"ud-w-{ud_pos[rid]}",
                        )
                    )
        elif str_roots and ud_roots:
            s_root = sorted(str_roots)[0]
            u_root = sorted(ud_roots)[0]
            if s_root in str_pos and u_root in ud_pos:
                lines.append(
                    _guarded_draw_line(
                        _STATUS_STYLE["unresolved"],
                        f"str-w-{str_pos[s_root]}",
                        f"ud-w-{ud_pos[u_root]}",
                    )
                )
    lines.append(r"\end{tikzpicture}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone document wrapper
# ---------------------------------------------------------------------------

_PREAMBLE = r"""
\documentclass[border=6pt]{article}
\usepackage{iftex}
\ifPDFTeX
  \usepackage[utf8]{inputenc}
  \usepackage[T2A]{fontenc}
  \usepackage[russian,english]{babel}
\fi
\usepackage{tikz}
\usepackage{tikz-dependency}
\usetikzlibrary{arrows, calc, positioning}
\begin{document}
"""

_POSTAMBLE = r"""
\end{document}
"""


def render_document(
    str_df: pd.DataFrame,
    ud_df: pd.DataFrame,
    sent_id: SentID,
    corr: EdgeCorrespondence,
    *,
    title: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Produce a compilable standalone .tex document containing the
    side-by-side picture for one sentence.

    The `overlay, remember picture` mechanism requires the two
    `dependency` environments to be wrapped so their nodes persist
    across blocks. We give them prefixed names via a small patch:
    `dep-word-row-1-col-N` nodes are already created by tikz-dependency
    under the labels `\\wordref{row}{col}`; we wrap them in an outer
    scope that we name manually to use `str-w-K` / `ud-w-K`.
    """
    body = render_pair(str_df, ud_df, sent_id, corr, **kwargs)
    title_line = f"\n\\par\\noindent\\textbf{{{_escape_latex(title)}}}\\par\\medskip\n" if title else ""
    return (
        _PREAMBLE
        + title_line
        + "\n% NOTE: connector lines are drawn only when alias nodes exist.\n\n"
        + body
        + _POSTAMBLE
    )


# ---------------------------------------------------------------------------
# Batch rendering for curator's 5 categories
# ---------------------------------------------------------------------------

def render_category_album(
    str_df: pd.DataFrame,
    ud_df: pd.DataFrame,
    corr_strict: EdgeCorrespondence,
    corr_extended: EdgeCorrespondence,
    picks: dict[int, Iterable[SentID]],
    *,
    out_prefix: str = "example",
) -> list[tuple[int, SentID, str]]:
    """
    Render one TikZ document per picked sentence, annotated with its
    category. Returns a list of (category, sent_id, latex_source)
    tuples for the caller to write to disk or embed.

    Categories 1-2 are rendered with `corr_strict`; categories 3-5 with
    `corr_extended` so that the visible correspondence lines reflect
    the algorithm that was supposed to cover the sentence.
    """
    out: list[tuple[int, SentID, str]] = []
    for cat, sids in picks.items():
        corr = corr_strict if cat in (1, 2) else corr_extended
        for sid in sids:
            title = f"Category {cat}. sent_id={sid}"
            tex = render_document(
                str_df, ud_df, sid, corr, title=title, show_unresolved=True
            )
            out.append((cat, sid, tex))
    return out
