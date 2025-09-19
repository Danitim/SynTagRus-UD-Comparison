import pathlib
import pyconll
from pyconll.unit.token import Token
from string import punctuation
import pathlib
import xml.etree.ElementTree as ET
import re
from collections import defaultdict

from convert.utils import get_ud_source, get_str_source
from convert.utils import restore_ellipsis

def convert_ud_to_ud(read_path, save_path):
    """
    Convert extended UD format to basic UD with explicit empty (ellipsis) nodes.
    """
    print("Converting extended UD to base UD...")

    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    for file_path in pathlib.Path(read_path).rglob("**/*.conllu"):
        conll = pyconll.load_from_file(file_path)

        for sent_idx, sent in enumerate(conll, start=1):
            id_map = {'0': '0'}
            dependents_by_ellipsis: dict[str, list[tuple[Token, str]]] = {}
            ellipsis_count = 0

            for new_id, tok in enumerate(sent, start=1):
                if tok.deps:
                    for head_id, rels in tok.deps.items():
                        if '.' in head_id:
                            if head_id not in dependents_by_ellipsis:
                                dependents_by_ellipsis[head_id] = []
                            dependents_by_ellipsis[head_id].append((tok, rels[0]))

                if tok.is_empty_node():
                    ellipsis_count += 1
                    dependents_by_ellipsis.setdefault(tok.id, [])

                id_map[tok.id] = str(new_id)

            if ellipsis_count:
                restore_ellipsis(sent, dependents_by_ellipsis)

            for new_id, tok in enumerate(sent, start=1):
                tok.head = str(id_map.get(tok.head, tok.head))
                tok.id = str(new_id)
                tok.deps = {}

            source, sid = get_ud_source(sent)
            sent.set_meta('sent_id', f"{sid}_{source}")

        save_file_path = pathlib.Path(save_path) / file_path.name
        with open(save_file_path, 'w', encoding='utf-8') as f:
            conll.write(f)

    print("Conversion completed.")

            
            
def convert_str_to_ud(read_path, save_path):
    """
    Конвертация корпуса СинТагРус в формат UD.
    """
    out_root = pathlib.Path(save_path)
    out_root.mkdir(parents=True, exist_ok=True)

    def _iter_visible_punct(s: str):
        if not s:
            return
        for ch in s:
            if ch and ch.strip():
                yield ch

    def _word_form(w_el) -> str:
        txt = (w_el.text or '').strip()
        return '_' if (not txt or txt == 'FANTOM') else txt
    
    def _word_lemma(w_el) -> str:
        lem = (w_el.get('LEMMA') or '').strip().lower()
        return lem if lem else '_'

    _RU2UD = {
        "1-Л": ("Лицо", "1-Л"),
        "2-Л": ("Лицо", "2-Л"),
        "3-Л": ("Лицо", "3-Л"),

        "ЕД": ("Число", "ЕД"),
        "МН": ("Число", "МН"),

        "МУЖ": ("Род", "МУЖ"),
        "ЖЕН": ("Род", "ЖЕН"),
        "СРЕД": ("Род", "СРЕД"),

        "ИМ": ("Падеж", "ИМ"),
        "РОД": ("Падеж", "РОД"),
        "ПАРТ": ("Падеж", "ПАРТ"),
        "ДАТ": ("Падеж", "ДАТ"),
        "ВИН": ("Падеж", "ВИН"),
        "ТВОР": ("Падеж", "ТВОР"),
        "ПР": ("Падеж", "ПР"),
        "МЕСТН": ("Падеж", "МЕСТН"),
        "ЗВ": ("Падеж", "ЗВ"),

        "ОД": ("Одуш", "ОД"),
        "НЕОД": ("Одуш", "НЕОД"),

        "СОВ": ("Вид", "СОВ"),
        "НЕСОВ": ("Вид", "НЕСОВ"),

        "НАСТ": ("Время", "НАСТ"),
        "ПРОШ": ("Время", "ПРОШ"),
        "НЕПРОШ": ("Время", "НЕПРОШ"),

        "ИЗЪЯВ": ("Накл", "ИЗЪЯВ"),
        "ПОВ": ("Накл", "ПОВ"),

        "ИНФ": ("Репр", "ИНФ"),
        "ПРИЧ": ("Репр", "ПРИЧ"),
        "ДЕЕПР": ("Репр", "ДЕЕПР"),

        "КР": ("Крат", "КР"),

        "СРАВ": ("СтепСравн", "СРАВ"),
        "ПРЕВ": ("СтепСравн", "ПРЕВ"),

        "СТРАД": ("Залог", "СТРАД"),

        # специальные теги -> misc
        "СЛ": "MISC:Служебное=Да",
        "НЕСТАНД": "MISC:Нестандарт=Да",
        "СМЯГ": "MISC:Смягч=Да",
        "НЕПРАВ": "MISC:Неправ=Да",
        "МЕТА": "MISC:Мета=Да",
    }


    def _parse_upos_feats_misc(w_el):
        raw = (w_el.get('FEAT') or '').strip()
        tokens = [t for t in re.split(r'[|\s]+', raw) if t]
        if not tokens:
            return '_', {}, '_'

        if tokens[0].upper().startswith("NID") and len(tokens) >= 2:
            cand = tokens[1].upper()
            if re.fullmatch(r'[A-Za-z]+', cand):
                upos = f"{tokens[0]}+{tokens[1]}"
                rest_tokens = tokens[2:]
            else:
                upos = tokens[0]
                rest_tokens = tokens[1:]
        else:
            upos = tokens[0]
            rest_tokens = tokens[1:]

        tail = ' '.join(rest_tokens)

        feats_map = defaultdict(set)
        misc_items = []

        if not tail:
            return (upos or '_'), {}, '_'

        grams = [g for g in re.split(r'[|\s]+', tail) if g]
        for g in grams:
            m = _RU2UD.get(g)
            if not m:
                continue
            if isinstance(m, tuple):
                k, v = m
                feats_map[k].add(v)
            else:
                # "MISC:Key=Value"
                _, kv = m.split(':', 1)
                misc_items.append(kv)

        if not feats_map and not misc_items and grams:
            misc_items.append('StrGram=' + ','.join(grams))

        misc = '|'.join(sorted(set(misc_items))) if misc_items else '_'
        if ('S' in misc and 'Softened' not in misc) or 'NUM' in misc:
            print(w_el.get('ID'), raw, misc)
                
        return (upos or '_'), dict(feats_map), misc

    def _word_id(w_el) -> str:
        wid = (w_el.get('ID') or '').strip()
        return wid if wid else str(id(w_el))

    def _word_dom_id(w_el):
        dom = (w_el.get('DOM') or '').strip()
        if not dom or dom == '_root' or dom == '0':
            return None
        return dom

    def _word_deprel(w_el) -> str:
        return (w_el.get('LINK') or '_')

    for file_path in pathlib.Path(read_path).rglob("**/*.tgt"):
        tree = ET.parse(file_path)
        root = tree.getroot()
        source = get_str_source(file_path)

        conll_chunks = []

        for body in root.findall('body'):
            for sent_idx, sent in enumerate(body.findall('S'), start=1):
                wlist = sent.findall('W')
                if not wlist:
                    continue

                seq = []

                for ch in _iter_visible_punct((sent.text or '').strip()):
                    seq.append(('punct', ch))

                for w in wlist:
                    seq.append(('word', w))
                    for ch in _iter_visible_punct((w.tail or '').strip()):
                        seq.append(('punct', ch))

                if seq and seq[-1][0] == 'word':
                    last_w = seq[-1][1]
                    lf = (last_w.text or '')
                    if lf.endswith('.'):
                        last_w.text = lf[:-1] if lf[:-1] else '_'
                        seq.append(('punct', '.'))

                wid_to_tokenid: dict[str, str] = {}
                token_id = 0
                for kind, obj in seq:
                    token_id += 1
                    if kind == 'word':
                        wid_to_tokenid[_word_id(obj)] = str(token_id)

                lines = []
                token_id = 0
                for kind, obj in seq:
                    token_id += 1
                    if kind == 'punct':
                        ch = obj
                        parts = [
                            str(token_id),  # id
                            ch,             # form
                            '_',            # lemma
                            'PUNCT',        # upos
                            '_',            # xpos
                            '_',            # feats
                            '_',            # head
                            '_',            # deprel
                            '_',            # deps
                            '_'             # misc
                        ]
                        lines.append('\t'.join(parts))
                    else:
                        w_el = obj
                        wid = _word_id(w_el)
                        form = _word_form(w_el)
                        lemma = _word_lemma(w_el)
                        upos, feats, misc = _parse_upos_feats_misc(w_el)
                        dom_id = _word_dom_id(w_el)
                        deprel = _word_deprel(w_el) or '_'

                        if dom_id is None:
                            head = '0'
                            deprel_out = 'root'
                        else:
                            head = wid_to_tokenid.get(dom_id, '0')
                            deprel_out = deprel
                            if head == str(token_id):
                                head = '0'
                                deprel_out = 'root'

                        feats_str = '_' if not feats else '|'.join(
                            f"{k}={','.join(sorted(vs))}" for k, vs in sorted(feats.items())
                        )

                        parts = [
                            str(token_id),  # id
                            form,           # form
                            lemma,          # lemma
                            upos,           # upos
                            '_',            # xpos
                            feats_str,      # feats
                            head,           # head
                            deprel_out,     # deprel
                            '_',            # deps
                            misc            # misc
                        ]
                        lines.append('\t'.join(parts))

                text_line = ' '.join(p.split('\t')[1] for p in lines if p.split('\t')[1] != '_')
                header = [
                    f"# sent_id = {sent_idx}_{source}",
                    f"# text = {text_line}",
                ]
                conll = '\n'.join(header + lines) + '\n'
                conll_chunks.append(conll)

        rel = ''.join(file_path.parts[1:])
        rel = rel[: rel.rfind('.tgt')]
        out_path = out_root / (rel + '.conllu')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(conll_chunks))

    print("Conversion completed.")
