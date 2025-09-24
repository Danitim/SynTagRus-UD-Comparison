import copy
from pyconll.unit.sentence import Sentence
from pyconll.unit.token import Token
from typing import Dict, List


def _pick_rel(old: Token, new: Token, sent: Sentence) -> str:
    def _case(tok: Token) -> str:
        v = tok.feats.get("Case") if tok.feats else None
        if not v:
            return None
        return next(iter(v)) if isinstance(v, set) else str(v)

    def _verbform(tok: Token) -> str:
        v = tok.feats.get("VerbForm") if tok.feats else None
        if not v:
            return None
        return next(iter(v)) if isinstance(v, set) else str(v)

    def _voice(tok: Token) -> str:
        v = tok.feats.get("Залог") if tok.feats else None
        if not v:
            return None
        return next(iter(v)) if isinstance(v, set) else str(v)

    def _has_rel_pron(tok: Token) -> bool:
        for t in sent:
            if t.head == tok.id and (t.feats or {}).get("PronType") == {"Rel"}:
                return True
        return False
    
    def _children(tok: Token) -> List[Token]:
        return [t for t in sent if t.head == tok.id]
    
    def _has_case_child(tok: Token) -> bool:
        return any(ch.upos == "ADP" and (ch.deprel or "") == "case" for ch in _children(tok))
    
    def _sent_has_lemma(lemmas: set) -> bool:
        lems = { (t.lemma or t.form or "").lower() for t in sent }
        return any(l in lems for l in lemmas)
    
    
    old_upos, new_pos = (old.upos or ""), (new.upos or "")
    old_case = _case(old)
    old_verbf = _verbform(old)
    new_voice = _voice(old)
    new_lemma = new.lemma or ""
    
    if old_upos == "ADP":
        return "case"
    if old_upos == "PUNCT":
        return "punct"
    if old_upos == "CCONJ":
        return "cc"
    if old_upos == "SCONJ":
        return "mark"
    if old_upos == "PART":
        lem = (old.lemma or "").lower()
        if lem == "бы":
            return "aux"
        if lem == "было":
            return "advmod"
        if lem == "не":
            return "advmod"
        return "discourse"
    
    if old_upos in {"VERB", "AUX"}:
        if old_verbf == "Conv": 
            return "advcl"
        if old_verbf == "Inf": 
            if new_pos in {"V", "A"}:
                return "xcomp"
            if new_pos in {"S", "PRON", "NUM", "NID"}:
                return "acl"
            return "xcomp"
        if new_pos in {"S", "PRON", "NUM", "NID"}:
            return "acl:relcl" if _has_rel_pron(old) else "acl"
        if new_pos in {"V", "A", "ADV"}:
            return "advcl"
        return "advcl"
    
    if new_pos == "V" and old_upos in {"ADJ", "NOUN", "PROPN"} and old_case == "Ins" and \
       any(t in new_lemma for t in {"считать", "называть", "объявлять", "обозвать", "звать"}):
        return "xcomp"
    
    if new_pos in {"V", "A"} and old_upos in {"NOUN", "PROPN", "PRON", "ADJ", "DET"} and old_case == "Nom":
        return "nsubj:pass" if new_voice == "СТРАД" else "nsubj"

    if new_pos == "V" and new_lemma == "быть" and \
       old_upos in {"NOUN", "PROPN", "PRON", "ADJ", "DET"} and old_case == "Gen" and \
       _sent_has_lemma({"сколько", "столько", "много", "немного", "несколько"}):
        return "nsubj"

    if new_pos == "V" and old_upos in {"NOUN", "PROPN", "PRON", "ADJ", "DET"} and \
       old_case == "Acc" and not _has_case_child(old):
        return "obj"

    if new_pos == "V" and old_upos in {"NOUN", "PROPN", "PRON", "NUM", "ADJ", "DET", "SYM"}:
        return "obl"

    if new_pos == "V" and old_upos == "X":
        return "nsubj"
        
    if new_pos in {"S", "PRON", "NUM", "NID"}:
        if old_upos == "ADJ":
            return "amod"
        if old_upos == "NUM":
            return "nummod"
        if old_upos in {"NOUN", "PROPN", "PRON"}:
            return "nmod"
        if old_upos == "ADV":
            return "advmod"
        if old_upos in {"VERB", "AUX"}:
            return "acl" if not _has_rel_pron(old) else "acl:relcl"
        
    if new_pos == "ADV":
        if old_upos == "ADV":
            return "advmod"
        if old_upos in {"NOUN", "PROPN", "PRON", "NUM"}:
            return "obl"
        if old_upos in {"VERB", "AUX"}:
            return "advcl"
    if new_pos == "A":
        if old_upos in {"NOUN", "PROPN", "PRON", "NUM"}:
            return "obl"
        if old_upos == "ADV":
            return "advmod" 
        
    if old_upos == "ADV":
        return "advmod" if new_pos in {"V", "A", "ADV", "NOUN", "PROPN"} else "obl"
    
    return "TEMP"

def fix_multi_roots(ud_sent: Sentence, str_sent: Sentence,
                    ud_to_str_token: Dict[str, Token]) -> None:
    by_ud_id = {t.id: t for t in ud_sent}

    for t in ud_sent:
        if (t.deprel or "").upper() != "TEMP":
            continue
        hid = str(t.head)

        ell = copy.deepcopy(by_ud_id.get(hid)) # ud token with SynTagRus lemma, upos and feats
        ell_str = ud_to_str_token.get(hid)
        if not ell or not ell_str:
            continue

        ell.upos = getattr(ell_str, "upos", None)
        ell.lemma = getattr(ell_str, "lemma", None)
        ell.feats = dict(getattr(ell_str, "feats", {}) or {})

        new_rel = _pick_rel(t, ell, ud_sent)

        t.deprel = new_rel
        
        if t.deprel == "TEMP":
            with open("TEMP_deprel.log", "a", encoding="utf-8") as f:
                f.write(f"#sent_id =  {ud_sent.meta_value('sent_id')}\n")
                f.write(f"#text    =  {ud_sent.meta_value('text')}\n")
                f.write(f"Old root:   {t.id}\t{t.form}\t{t.upos}\t{t.feats}\n")
                f.write(f"Ellipsis:   {ell.id}\t{ell.lemma}\t{ell.upos}\t{ell.feats}\n")
                f.write(f"New deprel: {new_rel}\n")
                f.write("-" * 100 + "\n")
