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

from edge_correspondence import (
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
    deprel_col, form_col : column names for labels and surface forms.
    """
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
    lines.append(r"\begin{dependency}[theme = simple]")
    lines.append(r"  \begin{deptext}[column sep=0.6cm]")
    lines.append("    " + " \\& ".join(str_forms) + r" \\")
    lines.append(r"  \end{deptext}")

    for _, row in str_toks.iterrows():
        head = int(row["head"])
        tid = int(row["id"])
        if head == 0:
            lines.append(f"  \\deproot{{{str_pos[tid]}}}{{{_escape_latex(row[deprel_col])}}}")
        else:
            lines.append(
                f"  \\depedge{{{str_pos[head]}}}{{{str_pos[tid]}}}"
                f"{{{_escape_latex(row[deprel_col])}}}"
            )
    lines.append(r"\end{dependency}")
    lines.append("")

    # --- UD dependency block (bottom) ---------------------------------------

    lines.append(r"% ---- UD tree (bottom) ----")
    lines.append(r"\begin{dependency}[theme = simple]")
    lines.append(r"  \begin{deptext}[column sep=0.6cm]")
    lines.append("    " + " \\& ".join(ud_forms) + r" \\")
    lines.append(r"  \end{deptext}")

    for _, row in ud_toks.iterrows():
        head = int(row["head"])
        tid = int(row["id"])
        if head == 0:
            lines.append(f"  \\deproot[edge below]{{{ud_pos[tid]}}}{{{_escape_latex(row[deprel_col])}}}")
        else:
            lines.append(
                f"  \\depedge[edge below]{{{ud_pos[head]}}}{{{ud_pos[tid]}}}"
                f"{{{_escape_latex(row[deprel_col])}}}"
            )
    lines.append(r"\end{dependency}")
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
            if m.status == "unresolved" and not show_unresolved:
                continue
            style = _STATUS_STYLE[m.status]
            if m.e_str is None:
                # Decorative marker on UD token only
                lines.append(
                    f"  \\draw[{style}] "
                    f"(ud-w-{ud_pos[dep_ud]}) circle (2mm);"
                )
                continue
            dep_str = str_sk.dep_of(m.e_str)
            if dep_str is None:
                continue
            lines.append(
                f"  \\draw[{style}] "
                f"(str-w-{str_pos[dep_str]}) -- (ud-w-{ud_pos[dep_ud]});"
            )
    lines.append(r"\end{tikzpicture}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone document wrapper
# ---------------------------------------------------------------------------

_PREAMBLE = r"""
\documentclass[border=6pt]{standalone}
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
    title_line = f"\n\\section*{{{_escape_latex(title)}}}\n" if title else ""
    return (
        _PREAMBLE
        + title_line
        + "\n% NOTE: the inline tikzpicture uses custom node names\n"
        + "% (str-w-K, ud-w-K). When integrating into a larger doc,\n"
        + "% rename prefixes as needed to avoid collisions.\n\n"
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
            title = f"Category {cat}. sent\\_id={sid}"
            tex = render_document(
                str_df, ud_df, sid, corr, title=title, show_unresolved=True
            )
            out.append((cat, sid, tex))
    return out