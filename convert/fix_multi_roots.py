import copy
from pyconll.unit.sentence import Sentence
from pyconll.unit.token import Token
from typing import Dict

def _case_from_feats(feats):
    if not feats or feats == "_" or not isinstance(feats, dict):
        return None
    vals = feats.get("Case")
    if not vals:
        return None
    if isinstance(vals, (set, list, tuple)):
        return next(iter(vals)) if vals else None
    return str(vals)

def _pick_rel_for_merge(old_upos: str, old_case, new_upos: str) -> str:
    return "dep"

def fix_multi_roots(ud_sent: Sentence, str_sent: Sentence,
                    ud_to_str_token: Dict[str, Token]) -> None:
    by_ud_id = {t.id: t for t in ud_sent}

    for t in ud_sent:
        if (t.deprel or "").upper() != "TEMP":
            continue
        hid = str(t.head)

        ell_ud = copy.deepcopy(by_ud_id.get(hid))
        ell_str = ud_to_str_token.get(hid)
        if not ell_ud or not ell_str:
            continue

        ell_ud.upos  = ell_str.upos or ell_ud.upos
        ell_ud.lemma = getattr(ell_str, "lemma", None) or ell_ud.lemma
        ell_ud.feats = dict(getattr(ell_str, "feats", {}) or {}) or ell_ud.feats

        new_upos  = ell_str.upos or ell_ud.upos or "ELLIPSIS"
        old_upos  = t.upos or "_"
        old_case  = _case_from_feats(t.feats)
        new_rel   = _pick_rel_for_merge(old_upos, old_case, new_upos)

        t.deprel = new_rel
        
        with open("TEMP_deprel.log", "a", encoding="utf-8") as f:
            f.write(f"#sent_id = {ud_sent.meta_value('sent_id')}\n")
            f.write(f"#text    = {ud_sent.meta_value('text')}\n")
            f.write(f"Old root: {t.id}\t{t.form}\t{old_upos}\t{t.feats}\n")
            f.write(f"Ellipsis: {ell_str.id}\t{ell_str.lemma}\t{new_upos}\t{ell_str.feats}\n")
            f.write("-" * 100 + "\n")
