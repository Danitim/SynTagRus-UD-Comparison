from typing import Dict, List, Optional
from pyconll.unit.sentence import Sentence
from pyconll.unit.token import Token
import re

from convert.ellipsis import reconcile_ellipsis_by_syntax
from convert.promote_ne4 import promote_ne4_constructions_in_str

_DASHES = "-–—"
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)
_ROMAN_RE = re.compile(r'^[IVXLCDM]+$', re.IGNORECASE)

_LAT_QUOTES = '\"\''
_CYR_QUOTES = '«»„“”‚’'
_EXTRA_PUNCT = ',…'

_EDGE_QUOTES = set('"\'«»„“”‚’‹›„‟')

def _strip_edge_quotes(s: str) -> str:
    if not s:
        return s
    i, j = 0, len(s) - 1
    while i <= j and s[i] in _EDGE_QUOTES:
        i += 1
    while j >= i and s[j] in _EDGE_QUOTES:
        j -= 1
    return s[i:j+1]

def _strip_quotes_commas_ellipsis(s: str) -> str:
    for ch in _CYR_QUOTES + _LAT_QUOTES + _EXTRA_PUNCT:
        s = s.replace(ch, '')
    return s

def _normalize(s: Optional[str]) -> str:
    if not s:
        return ""
    x = _strip_quotes_commas_ellipsis(s)
    x = x.lower()
    x = _WS_RE.sub("", x)
    for ch in _DASHES:
        x = x.replace(ch, "")
    return x

def _normalize_no_dots(s: Optional[str]) -> str:
    return _normalize(s).replace(".", "")

def _has_hyphen(s: Optional[str]) -> bool:
    if not s:
        return False
    return any(h in s for h in _DASHES)

def _is_punct(tok: Token) -> bool:
    form = (tok.form or '').strip()
    upos = (tok.upos or '').upper()
    feats = tok.feats or ''
    if form.lower() == 'x':
        return False
    if form == '×':
        return True
    if 'NumForm=Roman' in feats or _ROMAN_RE.fullmatch(form):
        return False
    return upos == 'PUNCT'

def _is_ellipsis(tok: Token) -> bool:
    return (tok.form is None or tok.form == "_") and not _is_punct(tok)

def _is_plus_token(tok: Token) -> bool:
    return (tok.form or '').strip() in {'+', '＋', '±'}

def _text_of(sent: Sentence) -> str:
    return " ".join([
        t.form if (t.form and t.form != '' and t.form != '_') else '<X>'
        for t in sent
    ])

def _log_unmatched(str_sent: Sentence, ud_sent: Sentence, reason: str):
    try:
        with open("unmatched.txt", "a", encoding="utf-8") as f:
            sid = str_sent.meta_value('sent_id') or ud_sent.meta_value('sent_id') or ''
            f.write(f"sent_id = {sid}\n")
            f.write(f"UD:  {_text_of(ud_sent)}\n")
            f.write(f"STR: {_text_of(str_sent)}\n")
            f.write(f"Mismatch: {reason}\n")
            f.write("-" * 60 + "\n")
    except Exception:
        pass

def _eq_forms(a: Optional[str], b: Optional[str]) -> bool:
    def _canon(x: Optional[str]) -> str:
        if x is None or x == '_':
            return '_'
        return _strip_edge_quotes(x)

    a_c, b_c = _canon(a), _canon(b)
    small_pairs = {('с','со'), ('со','с'), ('к','ко'), ('ко','к')}
    if (a_c, b_c) in small_pairs:
        return True
    a_nd = _normalize_no_dots(a_c)
    b_nd = _normalize_no_dots(b_c)
    abbr_eq = {('е', 'есть'), ('к', 'как')}
    if (a_nd, b_nd) in abbr_eq or (b_nd, a_nd) in abbr_eq:
        return True
    return (
        _normalize(a_c) == _normalize(b_c)
        or _normalize_no_dots(a_c) == _normalize_no_dots(b_c)
    )

def _build_deps(sent: Sentence) -> Dict[str, List[Token]]:
    deps: Dict[str, List[Token]] = {}
    for t in sent:
        if t.head not in (None, '_'):
            deps.setdefault(str(t.head), []).append(t)
    return deps

def _reindex_sentence(tokens: List[Token], orig_sent: Sentence) -> Sentence:
    old2new: Dict[str, str] = {}
    for new_i, t in enumerate(tokens, start=1):
        old2new[str(t.id)] = str(new_i)
    for new_i, t in enumerate(tokens, start=1):
        t.id = str(new_i)
        h = str(t.head) if t.head not in (None, '_') else '0'
        t.head = old2new.get(h, '0') if h != '0' else '0'
    sid = orig_sent.meta_value('sent_id') or ""
    text = " ".join([t.form for t in tokens if t.form and t.form != '_'])
    meta = []
    if sid:
        meta.append(f"# sent_id = {sid}")
    meta.append(f"# text = {text}")
    body = "\n".join(t.conll() for t in tokens)
    return Sentence("\n".join(meta + [body]))

def _edge_clean(s: Optional[str]) -> str:
    return _strip_edge_quotes(s or "")

def _eq_concat_either_order(a: Optional[str], b: Optional[str], u: Optional[str]) -> bool:
    if u is None:
        return False
    if _eq_forms((a or "") + (b or ""), u):
        return True
    if _eq_forms((b or "") + (a or ""), u):
        return True
    return False

def _detach_edge_quotes_in_str(str_sent: Sentence) -> Sentence:
    def _mk(form: str, upos: str, head: str, deprel: str,
            lemma=None, xpos=None, feats=None, id_value=None) -> Token:
        tok = Token(f"1\t{form or '_'}\t_\t{upos or '_'}\t_\t_\t{head or '0'}\t{deprel or 'punct'}\t_\t_")
        tok.lemma = lemma
        tok.xpos = xpos
        tok.feats = dict(feats) if feats else {}
        if id_value is not None:
            tok.id = str(id_value)
        return tok

    new_tokens: List[Token] = []
    qcnt = 0
    for t in str_sent:
        form = t.form or ''
        if not form:
            new_tokens.append(_mk(form, t.upos,
                                  str(t.head) if t.head not in (None, '_') else '0', t.deprel,
                                  lemma=t.lemma, xpos=t.xpos, feats=t.feats,
                                  id_value=t.id))
            continue
        i, j = 0, len(form) - 1
        leading, trailing = [], []
        while i <= j and form[i] in _EDGE_QUOTES:
            leading.append(form[i]); i += 1
        while j >= i and form[j] in _EDGE_QUOTES:
            trailing.append(form[j]); j -= 1
        core = form[i:j+1]
        if not leading and not trailing:
            new_tokens.append(_mk(form, t.upos,
                                  str(t.head) if t.head not in (None, '_') else '0', t.deprel,
                                  lemma=t.lemma, xpos=t.xpos, feats=t.feats,
                                  id_value=t.id))
            continue
        for q in leading:
            qcnt += 1
            new_tokens.append(_mk(q, 'PUNCT', '0', 'punct', id_value=f"q{qcnt}"))
        if core:
            new_tokens.append(_mk(core, t.upos,
                                  str(t.head) if t.head not in (None, '_') else '0', t.deprel,
                                  lemma=t.lemma, xpos=t.xpos, feats=t.feats,
                                  id_value=t.id))
        for q in trailing:
            qcnt += 1
            new_tokens.append(_mk(q, 'PUNCT', '0', 'punct', id_value=f"q{qcnt}"))
    return _reindex_sentence(new_tokens, str_sent)

def _extract_words_lc(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [w.lower() for w in _WORD_RE.findall(s)]

def _is_ud_glue(form: Optional[str]) -> bool:
    if not form:
        return False
    f = form
    return ('"' in f or '«' in f or '»' in f or ',' in f) and len(_extract_words_lc(f)) >= 2

def _multiset(lst: List[str]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for x in lst:
        d[x] = d.get(x, 0) + 1
    return d

def normalize_str_sentence_to_ud(ud_sent: Sentence, str_sent: Sentence) -> Optional[Sentence]:
    sid_ud = (ud_sent.meta_value('sent_id') or "").strip()
    str_sent = _detach_edge_quotes_in_str(str_sent)
    str_sent = promote_ne4_constructions_in_str(str_sent, ud_sent)
    str_sent = reconcile_ellipsis_by_syntax(ud_sent, str_sent)

    ud_core: List[Token]  = [t for t in ud_sent  if not _is_punct(t)]
    str_core: List[Token] = [t for t in str_sent if not _is_punct(t)]

    aligned_pairs: List[Token] = []
    link_map: Dict[str, str] = {"0": "0"}

    i = j = 0
    while j < len(ud_core):
        if i >= len(str_core):
            _log_unmatched(str_sent, ud_sent, f"STR core exhausted at UD index {j}")
            return None
        s1, u1 = str_core[i], ud_core[j]
        if _is_plus_token(s1) and not _eq_forms(s1.form, u1.form):
            i += 1
            continue
        if _is_ellipsis(u1):
            if _is_ellipsis(s1):
                aligned_pairs.append(s1)
                link_map[str(s1.id)] = str(u1.id)
                i += 1; j += 1
                continue
            _log_unmatched(str_sent, ud_sent,
                           f"Cannot align UD ellipsis at {j} with STR {i} ('{s1.form}')")
            return None
        if _eq_forms(_edge_clean(s1.form), _edge_clean(u1.form)):
            aligned_pairs.append(s1)
            link_map[str(s1.id)] = str(u1.id)
            i += 1; j += 1
            continue
        if i + 1 < len(str_core) and not _is_ellipsis(s1) and not _is_ellipsis(str_core[i+1]):
            s2 = str_core[i+1]
            if _eq_forms((_edge_clean(s1.form) or "") + (_edge_clean(s2.form) or ""),
                         _edge_clean(u1.form)):
                aligned_pairs.append(s2)
                link_map[str(s1.id)] = str(u1.id)
                link_map[str(s2.id)] = str(u1.id)
                i += 2; j += 1
                continue
            if _has_hyphen(u1.form) and _eq_concat_either_order(_edge_clean(s1.form),
                                                                _edge_clean(s2.form),
                                                                _edge_clean(u1.form)):
                aligned_pairs.append(s2)
                link_map[str(s1.id)] = str(u1.id)
                link_map[str(s2.id)] = str(u1.id)
                i += 2; j += 1
                continue
        if i + 2 < len(str_core) and not any(_is_ellipsis(x) for x in str_core[i:i+3]):
            s2, s3 = str_core[i+1], str_core[i+2]
            if _eq_forms((_edge_clean(s1.form) or "") + (_edge_clean(s2.form) or "") + (_edge_clean(s3.form) or ""),
                         _edge_clean(u1.form)):
                aligned_pairs.append(s3)
                link_map[str(s1.id)] = str(u1.id)
                link_map[str(s2.id)] = str(u1.id)
                link_map[str(s3.id)] = str(u1.id)
                i += 3; j += 1
                continue
        if _is_ud_glue(u1.form):
            ud_words_ms = _multiset(_extract_words_lc(u1.form))
            matched = False
            for win in range(5, 1, -1):
                if i + win - 1 >= len(str_core):
                    continue
                window = str_core[i:i+win]
                if any(_is_ellipsis(x) for x in window):
                    continue
                str_words = []
                for t in window:
                    str_words.extend(_extract_words_lc(t.form))
                if _multiset(str_words) == ud_words_ms:
                    rep = window[-1]
                    aligned_pairs.append(rep)
                    for t in window:
                        link_map[str(t.id)] = str(u1.id)
                    i += win; j += 1
                    matched = True
                    break
            if matched:
                continue
        moved = False
        WINDOW = 6
        for k in range(1, min(WINDOW, len(str_core) - i)):
            if _is_ellipsis(str_core[i+k]):
                break
            if _eq_forms(str_core[i + k].form, u1.form):
                tok = str_core.pop(i + k)
                str_core.insert(i, tok)
                moved = True
                break
        if moved:
            continue
        _log_unmatched(str_sent, ud_sent,
                       f"Cannot align UD token {j} ('{u1.form}') with STR {i} ('{s1.form}')")
        return None

    if i != len(str_core):
        _log_unmatched(str_sent, ud_sent, f"Extra STR core tokens remain: used {i} of {len(str_core)}")
        return None

    new_sent = Sentence(ud_sent.conll())

    def _mk_row_word(form: str, upos: str, head_core_idx: Optional[int], deprel: str, ud_id: str) -> dict:
        return {
            "kind": "word",
            "form": form or "_",
            "upos": upos or "_",
            "head_core_idx": head_core_idx,
            "deprel": (deprel or "_"),
            "_ud_id": ud_id,
            "_new_id": None,
        }

    def _mk_row_punct(form: str, anchor_core_idx: Optional[int], ud_head_udid: str) -> dict:
        return {
            "kind": "punct",
            "form": form or "_",
            "upos": "PUNCT",
            "anchor_core_idx": anchor_core_idx,
            "deprel": "punct",
            "_ud_head_udid": ud_head_udid,
            "_new_id": None,
        }

    ud_tokens: List[Token] = list(ud_sent)
    ud_core_positions = [i for i,t in enumerate(ud_tokens) if not _is_punct(t)]
    ud_core_tokens = [ud_tokens[i] for i in ud_core_positions]

    udid_to_core_idx: Dict[str,int] = {t.id: k for k,t in enumerate(ud_core_tokens)}

    ud_tokens_all = list(ud_sent)
    udid_to_ud_headid = {
        t.id: (str(t.head) if t.head not in (None, "", "_") else "0")
        for t in ud_tokens_all
    }
    udid_to_ud_deprel = {t.id: (t.deprel or "_") for t in ud_tokens_all}

    strid_to_core_idx_all: Dict[str,int] = {}
    for str_id, ud_id in link_map.items():
        if ud_id == "0":
            continue
        if ud_id in udid_to_core_idx:
            strid_to_core_idx_all[str_id] = udid_to_core_idx[ud_id]

    strid_to_headid: Dict[str, str] = {}
    for t in str_sent:
        sid = str(t.id)
        hid = str(t.head) if t.head not in (None, "", "_") else "0"
        strid_to_headid[sid] = hid

    if len(ud_core_tokens) != len(aligned_pairs):
        _log_unmatched(str_sent, ud_sent, "Aligned pairs length mismatch")
        return None

    core_rows: List[dict] = []
    strid_to_core_idx: Dict[str,int] = {}
    for k, (u_tok, s_tok) in enumerate(zip(ud_core_tokens, aligned_pairs)):
        core_rows.append(_mk_row_word(
            form = u_tok.form,
            upos = s_tok.upos,
            head_core_idx = None,
            deprel = s_tok.deprel,
            ud_id = u_tok.id
        ))
        strid_to_core_idx[str(s_tok.id)] = k

    for k, (u_tok, s_tok) in enumerate(zip(ud_core_tokens, aligned_pairs)):
        ud_here = u_tok.id
        group_str_ids = [sid for sid, uid in link_map.items() if uid == ud_here]
        cur = str(s_tok.head) if s_tok.head not in (None, "", "_") else "0"
        visited: set[str] = set()
        found_core: Optional[int] = None
        while cur != "0" and cur not in visited:
            visited.add(cur)
            if cur in group_str_ids:
                cur = strid_to_headid.get(cur, "0")
                continue
            if cur in strid_to_core_idx_all:
                cand_core = strid_to_core_idx_all[cur]
                if cand_core != k:
                    found_core = cand_core
                    break
                cur = strid_to_headid.get(cur, "0")
                continue
            cur = strid_to_headid.get(cur, "0")
        if found_core is None:
            if (s_tok.form or "").strip().lower() == "x":
                u_id = link_map.get(str(s_tok.id)) if "link_map" in locals() else None
                if u_id:
                    ud_parent = udid_to_ud_headid.get(u_id, "0")
                    if ud_parent != "0" and ud_parent in udid_to_core_idx:
                        core_rows[k]["head_core_idx"] = udid_to_core_idx[ud_parent]
                        core_rows[k]["deprel"] = udid_to_ud_deprel.get(u_id, core_rows[k]["deprel"] or "dep")
                    else:
                        core_rows[k]["head_core_idx"] = None
                        core_rows[k]["deprel"] = "root"
                else:
                    core_rows[k]["head_core_idx"] = None
                    core_rows[k]["deprel"] = "root"
            else:
                core_rows[k]["head_core_idx"] = None
                core_rows[k]["deprel"] = "root"
        else:
            core_rows[k]["head_core_idx"] = found_core
            if (core_rows[k]["deprel"] or "").lower() == "root":
                core_rows[k]["deprel"] = "dep"

    out_rows: List[dict] = []
    core_cursor = 0
    udid_to_out_index: Dict[str,int] = {}

    def _emit_punct_from_ud(u_punct: Token):
        ud_head = str(u_punct.head) if u_punct.head not in (None,'_') else '0'
        if ud_head != '0' and ud_head in udid_to_core_idx:
            anchor_core = udid_to_core_idx[ud_head]
        else:
            anchor_core = None
        out_rows.append(_mk_row_punct(u_punct.form, anchor_core, ud_head))

    for i, u in enumerate(ud_tokens):
        if _is_punct(u):
            _emit_punct_from_ud(u)
        else:
            row = core_rows[core_cursor]
            out_rows.append(dict(row))
            udid_to_out_index[u.id] = len(out_rows)-1
            core_cursor += 1

    for new_id, r in enumerate(out_rows, start=1):
        r["_new_id"] = str(new_id)

    def _coreidx_to_newid(core_idx: Optional[int]) -> Optional[str]:
        if core_idx is None:
            return None
        target_ud = core_rows[core_idx]["_ud_id"]
        out_pos = udid_to_out_index.get(target_ud)
        return (out_rows[out_pos]["_new_id"] if out_pos is not None else None)

    for r in out_rows:
        if r["kind"] != "word":
            continue
        if r["head_core_idx"] is None:
            r["head"] = '0'
            r["deprel"] = 'root'
        else:
            nid = _coreidx_to_newid(r["head_core_idx"])
            if nid is None or nid == r["_new_id"]:
                r["head"] = '0'
                r["deprel"] = 'root'
            else:
                r["head"] = nid
                if (r["deprel"] or '').lower() == 'root':
                    r["deprel"] = 'dep'

    for idx, r in enumerate(out_rows):
        if r["kind"] != "punct":
            continue
        nid = _coreidx_to_newid(r["anchor_core_idx"])
        if nid is None:
            anchor = None
            for j in range(idx+1, len(out_rows)):
                if out_rows[j]["kind"] == "word":
                    anchor = out_rows[j]["_new_id"]; break
            if anchor is None:
                for j in range(idx-1, -1, -1):
                    if out_rows[j]["kind"] == "word":
                        anchor = out_rows[j]["_new_id"]; break
            r["head"] = anchor or '0'
        else:
            r["head"] = nid
        if not r["deprel"] or r["deprel"].lower() == "root":
            r["deprel"] = "punct"

    sid = ud_sent.meta_value('sent_id') or ""
    text = " ".join([rr["form"] for rr in out_rows if rr["form"] != '_'])
    meta = []
    if sid:
        meta.append(f"# sent_id = {sid}")
    meta.append(f"# text = {text}")
    body = "\n".join(
        f"{rr['_new_id']}\t{rr['form']}\t_\t{rr['upos']}\t_\t_\t{rr['head']}\t{rr['deprel']}\t_\t_"
        for rr in out_rows
    )
    res = Sentence("\n".join(meta + [body]))
    return res
