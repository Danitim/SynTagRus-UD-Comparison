from __future__ import annotations
import pandas as pd


STR_KEY_MAP: dict[str, str] = {
    "Одуш":     "Animacy",
    "Род":      "Gender",
    "Число":    "Number",
    "Падеж":    "Case",
    "Вид":      "Aspect",
    "Время":    "Tense",
    "Залог":    "Voice",
    "Крат":     "Variant",
    "Лицо":     "Person",
    "Накл":     "Mood",
    "Репр":     "VerbForm",
    "СтепСравн":"Degree",
}

UD_COMMON_KEYS: frozenset[str] = frozenset(STR_KEY_MAP.values())
_UD_TO_STR_KEY: dict[str, str] = {v: k for k, v in STR_KEY_MAP.items()}

# §3.1: STR category name → UD category name (includes POS/UPOS)
CATEGORY_MAP: dict[str, str] = {**STR_KEY_MAP, "POS": "UPOS"}
_UD_TO_STR_FULL: dict[str, str] = {v: k for k, v in CATEGORY_MAP.items()}


# ─── Utility: feature parsing / translation ──────────────────────────────────

def _is_reflexive(lemma) -> bool:
    if pd.isna(lemma) or not lemma:
        return False
    s = str(lemma).lower()
    return s.endswith("ся") or s.endswith("сь")


def parse_feats(feats_str) -> dict[str, str]:
    """Parse feats string 'A=a|B=b' to dict. Returns {} for _ or NaN."""
    if pd.isna(feats_str) or feats_str in ("", "_"):
        return {}
    return dict(pair.split("=", 1) for pair in str(feats_str).split("|") if "=" in pair)


def restrict_ud_feats(feats_str) -> dict[str, str]:
    """Parse UD feats and restrict to common keys K_∩ (UD_COMMON_KEYS)."""
    return {k: v for k, v in parse_feats(feats_str).items() if k in UD_COMMON_KEYS}


def translate_str_feats(feats_str, upos, lemma) -> str:
    raw = "" if pd.isna(feats_str) or feats_str in ("", "_") else str(feats_str)

    parsed: dict[str, str] = {}
    for pair in raw.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            parsed[k] = v

    pairs: list[str] = []

    for str_key, ud_key in STR_KEY_MAP.items():
        if str_key in parsed:
            ud_val = STR_VALUE_MAP.get(str_key, {}).get(parsed[str_key], parsed[str_key])
            pairs.append(f"{ud_key}={ud_val}")

    if upos is not None:
        for str_key, ud_key, ud_val, applicable in _STR_ABSENT_DEFAULTS:
            if str_key not in parsed and upos in applicable:
                if str_key == "Залог" and upos == "AUX" and lemma in ("бы", "б"):
                    continue
                pairs.append(f"{ud_key}={ud_val}")

    if _is_reflexive(lemma) and any(pair.startswith("Voice") for pair in pairs):
        idx = next(i for i, pair in enumerate(pairs) if pair.startswith("Voice"))
        pairs[idx] = "Voice=Mid"

    return "|".join(sorted(pairs)) if pairs else "_"


def translate_str_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["feats_ud_g"] = [
        translate_str_feats(f, u, lem) for f, u, lem in zip(df["feats_g"], df["upos_g"], df["lemma"])
    ]
    out["feats_ud_p"] = [
        translate_str_feats(f, u, lem) for f, u, lem in zip(df["feats_p"], df["upos_p"], df["lemma"])
    ]
    return out


# ─── §4: Error sets ──────────────────────────────────────────────────────────

def _compute_E_str(upos_g, upos_p, feats_g, feats_p) -> frozenset[str]:
    """STR error set — uses STR category names; "POS" for part-of-speech."""
    g = parse_feats(feats_g)
    p = parse_feats(feats_p)
    errors: set[str] = {k for k in set(g) | set(p) if g.get(k) != p.get(k)}
    if upos_g != upos_p:
        errors.add("POS")
    return frozenset(errors)


def _compute_E_ud(upos_g, upos_p, feats_g, feats_p) -> frozenset[str]:
    """UD error set — uses UD category names; "UPOS" for part-of-speech."""
    g = parse_feats(feats_g)
    p = parse_feats(feats_p)
    errors: set[str] = {k for k in set(g) | set(p) if g.get(k) != p.get(k)}
    if upos_g != upos_p:
        errors.add("UPOS")
    return frozenset(errors)


# ─── §6: Build comparison df ─────────────────────────────────────────────────

def build_comparison_df(data: dict, split: str = "full") -> pd.DataFrame:
    """
    Merge STR and UD dataframes; compute E_str and E_ud for each token.

    E_str – frozenset of STR category names where M_str erred
    E_ud  – frozenset of UD category names where M_ud erred
    """
    str_key = "str" if split == "full" else f"str-{split}"
    ud_key  = "ud"  if split == "full" else f"ud-{split}"

    str_df = data[str_key].reset_index()
    ud_df  = data[ud_key].reset_index()

    merged = str_df.merge(ud_df, on=["sent_id", "id"], suffixes=("_str", "_ud"))

    merged["E_str"] = [
        _compute_E_str(ug, up, fg, fp)
        for ug, up, fg, fp in zip(
            merged["upos_g_str"], merged["upos_p_str"],
            merged["feats_g_str"], merged["feats_p_str"],
        )
    ]
    merged["E_ud"] = [
        _compute_E_ud(ug, up, fg, fp)
        for ug, up, fg, fp in zip(
            merged["upos_g_ud"], merged["upos_p_ud"],
            merged["feats_g_ud"], merged["feats_p_ud"],
        )
    ]
    return merged


# ─── §5: Error signatures (per-category filter) ───────────────────────────────

def compute_error_signatures(merged_df: pd.DataFrame, mode: str = "clean") -> pd.DataFrame:
    """
    §5: For each token, build triplets (c_ud, str_gold, ud_gold, ud_pred) for all
    (c_str, c_ud) ∈ CATEGORY_MAP satisfying the per-category condition (§5.2):

      mode="clean"   — STR correct AND UD wrong  (main table)
      mode="control" — STR wrong   AND UD wrong  (control table, §8.2)

    sig = sorted tuple of qualifying triplets. Tokens with no triplets are excluded.
    """
    if mode not in ("clean", "control"):
        raise ValueError(f"mode must be 'clean' or 'control', got {mode!r}")

    rows: list[dict] = []

    for _, row in merged_df.iterrows():
        str_gold_feats = parse_feats(row["feats_g_str"])
        ud_gold_feats  = parse_feats(row["feats_g_ud"])
        ud_pred_feats  = parse_feats(row["feats_p_ud"])
        E_str = row["E_str"]
        E_ud  = row["E_ud"]

        triplets: list[tuple] = []
        for c_str, c_ud in CATEGORY_MAP.items():
            # category must be present in both gold standards
            str_has = (c_str == "POS") or (c_str in str_gold_feats)
            ud_has  = (c_ud  == "UPOS") or (c_ud  in ud_gold_feats)
            if not (str_has and ud_has):
                continue

            str_correct = c_str not in E_str
            ud_error    = c_ud in E_ud

            include = (str_correct and ud_error) if mode == "clean" else ((not str_correct) and ud_error)
            if not include:
                continue

            str_g = (row["upos_g_str"] or "∅") if c_str == "POS" else str_gold_feats.get(c_str, "∅")
            ud_g  = (row["upos_g_ud"]  or "∅") if c_ud  == "UPOS" else ud_gold_feats.get(c_ud,  "∅")
            ud_p  = (row["upos_p_ud"]  or "∅") if c_ud  == "UPOS" else ud_pred_feats.get(c_ud,  "∅")
            triplets.append((c_ud, str_g, ud_g, ud_p))

        if not triplets:
            continue

        d = row.to_dict()
        d["sig"] = tuple(sorted(triplets))
        rows.append(d)

    cols = [*merged_df.columns, "sig"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# ─── §7: Grouping and ranking ────────────────────────────────────────────────

def aggregate_signatures(token_df: pd.DataFrame) -> pd.DataFrame:
    """§7.1: Group by sig; count n_s."""
    return (
        token_df.groupby("sig", sort=False)
        .agg(n=("sig", "count"))
        .reset_index()
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )


def _match_gold_pattern(
    str_feats: dict, str_upos: str,
    ud_feats: dict,  ud_upos: str,
    sig: tuple,
) -> bool:
    """True if token's gold annotation matches (str_g, ud_g) for every triplet in sig."""
    for c_ud, str_g, ud_g, _ in sig:
        c_str = _UD_TO_STR_FULL[c_ud]
        actual_str_g = (str_upos or "∅") if c_str == "POS" else str_feats.get(c_str, "∅")
        actual_ud_g  = (ud_upos  or "∅") if c_ud  == "UPOS" else ud_feats.get(c_ud,  "∅")
        if actual_str_g != str_g or actual_ud_g != ud_g:
            return False
    return True


def add_gold_pct(comparison_df: pd.DataFrame, sigs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pct = n / n_gold * 100, where n_gold is the number of tokens in the full
    comparison whose gold annotation matches the (str_g, ud_g) pattern of the signature.
    """
    str_feats = comparison_df["feats_g_str"].apply(parse_feats).tolist()
    str_upos  = comparison_df["upos_g_str"].tolist()
    ud_feats  = comparison_df["feats_g_ud"].apply(parse_feats).tolist()
    ud_upos   = comparison_df["upos_g_ud"].tolist()

    sigs_df = sigs_df.copy()
    n_gold = [
        sum(
            _match_gold_pattern(sf, su, uf, uu, sig)
            for sf, su, uf, uu in zip(str_feats, str_upos, ud_feats, ud_upos)
        )
        for sig in sigs_df["sig"]
    ]
    sigs_df["n_gold"] = n_gold
    sigs_df["pct"] = (
        sigs_df["n"] / sigs_df["n_gold"].replace(0, float("nan")) * 100
    ).round(1)
    return sigs_df


def rank_signatures(sigs_df: pd.DataFrame) -> pd.DataFrame:
    """Sort signatures by n descending."""
    return sigs_df.sort_values("n", ascending=False).reset_index(drop=True)


# ─── §7.2: Multi-split result table ──────────────────────────────────────────

def build_result_table(data: dict, filter=True, mode: str = "clean") -> pd.DataFrame:
    """
    §7.2: Build combined table with n and pct for full/old/new splits.

    mode: "clean"   — main table   (STR correct, UD wrong per category)
          "control" — control table (STR wrong,   UD wrong per category)
    """
    split_results: dict[str, dict] = {}
    for split in ["full", "old", "new"]:
        split_results[split] = analyze_conversion_errors(data, split, mode=mode)
    split_sigs: dict[str, pd.DataFrame] = {s: r["signatures"] for s, r in split_results.items()}

    all_sigs: set = set()
    for df in split_sigs.values():
        all_sigs.update(df["sig"])

    rows: list[dict] = []
    for sig in all_sigs:
        row: dict = {"sig": sig}
        for split in ["full", "old", "new"]:
            df = split_sigs[split]
            match = df[df["sig"] == sig]
            row[f"n_{split}"]   = int(match["n"].iloc[0])     if len(match) else 0
            row[f"pct_{split}"] = float(match["pct"].iloc[0]) if len(match) else 0.0
        rows.append(row)
        
    out = pd.DataFrame(rows)
    out["total"] = out["n_full"] + out["n_old"] + out["n_new"]
        
    if filter:
        mask_n  = (out["n_full"] > 10) | (out["n_old"] > 10) | (out["n_new"] > 10)
        mask_pct = (out["pct_full"] > 1.0) | (out["pct_old"] > 1.0) | (out["pct_new"] > 1.0)
        out = out[mask_n & mask_pct]
    
    out = out.sort_values("n_new", ascending=False).reset_index(drop=True)

    return out, split_results


# ─── Full pipeline ────────────────────────────────────────────────────────────

def analyze_conversion_errors(data: dict, split: str = "full", mode: str = "clean") -> dict:
    """
    Full pipeline (method.md §4-7).

    split: 'full' | 'old' | 'new'
    mode:  'clean'   — main table   (STR correct AND UD wrong, per category)
           'control' — control table (STR wrong   AND UD wrong, per category)

    Returns dict:
      comparison  – merged STR+UD df with E_str, E_ud
      tokens      – tokens with ≥1 qualifying triplet; column sig added
      signatures  – ranked sigs with n, n_gold, pct
    """
    cmp_df    = build_comparison_df(data, split)
    annotated = compute_error_signatures(cmp_df, mode=mode)
    sigs      = aggregate_signatures(annotated)
    sigs      = add_gold_pct(cmp_df, sigs)

    return {
        "comparison":  cmp_df,
        "tokens":      annotated,
        "signatures":  sigs,
    }
