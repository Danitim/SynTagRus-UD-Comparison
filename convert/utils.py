import pyconll
import pathlib

from pyconll.unit.sentence import Sentence
from pyconll.unit.token import Token
from string import punctuation


def get_ud_source(sent):
    
    source = sent.meta_value('sent_id')
    id = source[source.rfind('.xml')+5:]
    source = source[: source.rfind('.xml')]
    
    # while not source[-1].isalpha():
    #     source = source[:-1]

    return source, id


def get_str_source(file_path: pathlib.Path) -> str:
    parts = list(file_path.parts)
    
    if "SynTagRus" in parts:
        idx = parts.index("SynTagRus")
        parts = parts[idx+1:]
        
    source = "".join(parts)
    
    if source.endswith(".tgt"):
        source = source[:-4]
    
    return source


def get_source(sent):
    '''
    Extract source name from the sentence
    
    Parameters:
    sent (pyconll.unit.sentence.Sentence): sentence.
    
    Returns:
    source_name (str): source name of the sentence.
    '''
    first_underscore = sent.meta_value('sent_id').find('_')
    return sent.meta_value('sent_id')[first_underscore + 1:]


def search_source_name(source_name, path):
    '''
    Search for files with the given source name in the specified path.
    
    Parameters:
    source_name (str): source name to search for.
    path (str): path to the directory.
    
    Returns:
    list: list of file paths containing the source name.
    '''
    files = []
    for file_path in pathlib.Path(path).rglob("**/*.conllu"):
        if file_path.name.lower().find(source_name.lower()) != -1:
            files.append(file_path)
    return files

def match_sentences(ud_sent, str_sent):
    '''
    Compare two sentences for complete word form match without
        counting punctuation.
    
    Parameters:
    ud_sent (pyconll.unit.sentence.Sentence): UD sentence.
    str_sent (pyconll.unit.sentence.Sentence): SynTagRus sentence.
    
    Returns:
    str_sent (pyconll.unit.sentence.Sentence): SynTagRus sentence with
        fixed punctuation, if sentences match, None otherwise.
    '''
    def strip_punct(token):
        return token.form.strip(punctuation + '…').lower() if (token.form and token.form != '_') else '_'
    
    def set_form(token, form) -> Token:
        token_id = token.id
        upos = token.upos
        head, deprel= token.head, token.deprel
        
        return Token(f"{token_id}\t{form}\t_\t{upos}\t_\t_\t{head}\t{deprel}\t_\t_")
    
    
    ud_words = [token for token in ud_sent if token.upos != 'PUNCT']
    str_words = [token for token in str_sent if token.upos != 'PUNCT']
    
    if len(ud_words) != len(str_words):
        if len([word for word in ud_words if strip_punct(word) != '_']) != len([word for word in str_words if strip_punct(word) != '_']):
            with open("length_unmatched.txt", 'a', encoding='utf-8') as f:
                f.write("sent_id = " + str_sent.meta_value('sent_id') + '\n')
                f.write("UD: " + ' '.join([strip_punct(token) for token in ud_sent]) + '\n')
                f.write("STR: " + ' '.join([strip_punct(token) for token in str_sent]) + '\n')
                f.write("-" * 100 + '\n')
        return None
    
    unmatched = []
    for idx, (ud_word, str_word) in enumerate(zip(ud_words, str_words)):
        if strip_punct(ud_word) != strip_punct(str_word):
            str_words[idx] = set_form(str_word, ud_word.form)
            unmatched.append(strip_punct(ud_word) + '!=' + strip_punct(str_word))
            
    if len(unmatched) > 0:
        ud_word_forms = [strip_punct(token) for token in ud_sent]
        str_word_forms = [strip_punct(token) for token in str_sent]
        with open("unmatched.txt", 'a', encoding='utf-8') as f:
            f.write("sent_id = " + str_sent.meta_value('sent_id') + '\n')
            for unmatched_pair in unmatched:
                f.write(unmatched_pair + '\n')
            f.write("UD: " + ' '.join(ud_word_forms) + '\n')
            f.write("STR: " + ' '.join(str_word_forms) + '\n')
            f.write("-" * 100 + '\n')
        
    
    new_sent = Sentence(ud_sent.conll())
    str_index = 0
    link_map = {'0': '0'}
    for token in new_sent:
        if token.upos != 'PUNCT':
            str_token = str_words[str_index]
            str_index += 1
            
            link_map[str_token.id] = token.id
            
    str_index = 0
    for token in new_sent:
        if token.upos != 'PUNCT':
            str_token = str_words[str_index]
            str_index += 1
            
            token.upos = str_token.upos
            token.xpos = None
            token.head = link_map[str_token.head]
            token.deprel = str_token.deprel
            
        token.lemma = None
        token.feats = {}
        token.misc = {}
            
    return new_sent
    

import logging

# Разрешённые базовые отношения UD
BASIC_RELATIONS = {
    "acl", "acl:relcl", "advcl", "advmod", "amod", "appos",
    "aux", "aux:pass", "case", "cc", "ccomp", "compound", "conj",
    "cop", "csubj", "csubj:pass", "dep", "det", "discourse",
    "dislocated", "expl", "fixed", "flat", "flat:foreign",
    "flat:name", "iobj", "list", "mark", "nmod", "nsubj",
    "nsubj:outer", "nsubj:pass", "nummod", "nummod:entity",
    "nummod:gov", "obj", "obl", "obl:agent", "obl:depict",
    "obl:float", "obl:pronmod", "obl:tmod", "orphan", "parataxis",
    "parataxis:discourse", "punct", "root", "vocative", "xcomp",
}

def _rel_base(r):
    if r is None:
        return None
    s = str(r).strip()
    if not s or s == "_":
        return None
    return s.split(":", 1)[0]

def _id_key(tok_id):
    try:
        if "." in tok_id:
            a, b = tok_id.split(".", 1)
            return (int(a), int(b))
        return (int(tok_id), 0)
    except Exception:
        return (10**9, 10**9)

def restore_ellipsis(sent, _dependents_from_caller):
    by_id = {t.id: t for t in sent}
    sent_id_str = getattr(sent, "id", None) or getattr(sent, "meta", {}).get("sent_id") or (sent.meta_value("sent_id") if hasattr(sent, "meta_value") else None) or "?"

    main_root_id = None
    for t in sent:
        if (not t.is_empty_node()) and str(t.head) == "0":
            main_root_id = t.id
            break
    if main_root_id is None:
        non_empty_ids = [t.id for t in sent if not t.is_empty_node()]
        main_root_id = min(non_empty_ids, key=_id_key) if non_empty_ids else "1"

    tok_info = {}
    for tok in sent:
        if tok.is_empty_node():
            continue

        base_head = str(tok.head) if tok.head is not None else None
        base_rel = _rel_base(tok.deprel) or "dep"

        info = {"base_head": base_head, "base_rel": base_rel, "ellipses": [], "non_empty_deps": []}
        deps = tok.deps or {}

        for h, rels in deps.items():
            if not rels:
                continue
            for r in rels:
                base = _rel_base(r)
                if base is None:
                    continue
                if base not in BASIC_RELATIONS:
                    continue

                if base_head is not None and base == base_rel:
                    if h == base_head:
                        info["non_empty_deps"].append((h, base))
                        continue
                    else:
                        if "." in h:
                            info["ellipses"].append((h, base, r))
                            continue
                        else:
                            continue

                if "." in h:
                    info["ellipses"].append((h, base, r))
                else:
                    info["non_empty_deps"].append((h, base))

        info["ellipses"].sort(key=lambda x: _id_key(x[0]))
        info["non_empty_deps"].sort(key=lambda x: _id_key(x[0]))
        tok_info[tok.id] = info

    ellipsis_heads = {}
    for e in (t for t in sent if t.is_empty_node()):
        heads = set()
        deps_dict = e.deps or {}
        for h, rels in deps_dict.items():
            if h == "0":
                continue
            if not rels:
                continue
            for r in rels:
                base = _rel_base(r)
                if base is None or base not in BASIC_RELATIONS:
                    continue
                if h in by_id:
                    heads.add(h)
        ellipsis_heads[e.id] = heads
    
    token2ellipsis = {}
    for tok in sent:
        if tok.is_empty_node():
            continue
        info = tok_info[tok.id]
        if not info["ellipses"]:
            if info["base_rel"] == "orphan" and info["non_empty_deps"]:
                cands = info["non_empty_deps"]
                pick = next(((h, b) for (h, b) in cands if b != "case"), None) or cands[0]
                new_head, new_rel = pick
                tok.head = new_head
                tok.deprel = new_rel
                logging.info(f"[{sent_id_str}] replaced orphan on token {tok.id} -> {new_head}:{new_rel} from enhanced DEPS")
            else:
                ok = any((hid == info["base_head"] and base == info["base_rel"]) for (hid, base) in info["non_empty_deps"])
                if not ok and info["non_empty_deps"]:
                    logging.info(f"[{sent_id_str}] keep base {info['base_head']}:{info['base_rel']} for token {tok.id}; non-elliptic DEPS {info['non_empty_deps']} do not match base")
            continue


        ell = info["ellipses"]

        safe_ell = []
        for (eid, brel, _full) in ell:
            heads = ellipsis_heads.get(eid, set())
            if heads == {tok.id}:
                continue
            safe_ell.append((eid, brel, _full))

        if not safe_ell:
            continue

        matching = [p for p in safe_ell if p[1] == info["base_rel"]]
        pick = matching[0] if matching else safe_ell[0]
        eid, base_rel, _full = pick
        token2ellipsis[tok.id] = (eid, base_rel)


    empty_ids = sorted([t.id for t in sent if t.is_empty_node()], key=_id_key)
    ell_dependents = {eid: [] for eid in empty_ids}
    for tok_id, (eid, base_rel) in token2ellipsis.items():
        if eid in ell_dependents:
            ell_dependents[eid].append((by_id[tok_id], base_rel))

    for eid in empty_ids:
        E = by_id[eid]
        dep_list = ell_dependents.get(eid, [])
        future_dep_ids = {t.id for (t, _) in dep_list}

        candidates = []   # (head_id, base_rel)
        wants_root = False
        had_2cycle_skip = False
        deps_dict = E.deps or {}
        for h, rels in deps_dict.items():
            if not rels:
                continue
            for r in rels:
                base = _rel_base(r)
                if base is None:
                    continue
                if base not in BASIC_RELATIONS:
                    # logging.warning(f"[{sent_id_str}] drop non-basic rel at ellipsis {eid}->{h}:{r}")
                    continue
                if h == "0":
                    if base == "root":
                        wants_root = True
                    continue
                if h not in by_id:
                    logging.warning(f"[{sent_id_str}] unknown head for ellipsis {eid}->{h}:{r}")
                    continue
                if h in future_dep_ids:
                    had_2cycle_skip = True
                    continue
                candidates.append((h, base))

        chosen_head = None
        chosen_rel = None
        if wants_root and len(candidates) == 1:
            chosen_head, chosen_rel = candidates[0]
        elif wants_root and len(candidates) == 0:
            chosen_head, chosen_rel = "0", "root"
            main_root_id = E.id
        elif len(candidates) > 1:
            conj_ellipsis = next((c for c in candidates if "." in c[0] and c[1] == "conj"), None)
            if conj_ellipsis is not None:
                chosen_head, chosen_rel = conj_ellipsis
            else:
                any_ellipsis = sorted((c for c in candidates if "." in c[0]), key=lambda x: _id_key(x[0]))
                if any_ellipsis:
                    chosen_head, chosen_rel = any_ellipsis[0]
                else:
                    conj_normal = next((c for c in candidates if "." not in c[0] and c[1] == "conj"), None)
                    if conj_normal is not None:
                        chosen_head, chosen_rel = conj_normal
                    else:
                        # logging.warning(f"[{sent_id_str}] ellipsis {eid} has multiple heads {candidates}; no conj-head → picking deterministically")
                        candidates.sort(key=lambda x: _id_key(x[0]))
                        chosen_head, chosen_rel = candidates[0]
        elif len(candidates) == 1:
            chosen_head, chosen_rel = candidates[0]
        else:
            if had_2cycle_skip:
                logging.warning(f"[{sent_id_str}] ellipsis {eid} had only heads that create 2-cycle; attaching to main root")
            else:
                logging.warning(f"[{sent_id_str}] ellipsis {eid} has no valid heads; attach to main root")
            chosen_head, chosen_rel = main_root_id, "dep"

        E.head = chosen_head
        E.deprel = chosen_rel

        for tok, rel_base in dep_list:
            tok.head = E.id
            base_rel = rel_base or "dep"
            if base_rel not in BASIC_RELATIONS:
                base_rel = "dep"
            tok.deprel = base_rel
    
    
    roots = [tok for tok in sent if str(tok.head) == "0"]
    if len(roots) > 1:       
        ell_roots = [r for r in roots if "." in str(r.id)]
        non_ell_roots = [r for r in roots if "." not in str(r.id)]
        if ell_roots and non_ell_roots:
            ell = ell_roots[0]
            for tok in non_ell_roots:
                tok.head = ell.id
                tok.deprel = "TEMP"
         
        with open("uncertain.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"[{sent_id_str}] Multiple roots detected:\n")
            for root in roots:
                log_file.write(
                    f"  - Token ID: {root.id}, "
                    f"Form: {root.form}, "
                    f"UPOS: {root.upos}\n"
                )
            log_file.write("-" * 100 + "\n")

