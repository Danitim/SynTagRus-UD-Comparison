import pathlib
import pyconll
import xml.etree.ElementTree as ET
from pyconll.unit.token import Token
from string import punctuation

from convert.utils import get_ud_source, get_str_source
from convert.utils import restore_ellipsis

def convert_ud_to_ud(read_path, save_path):
    '''
    Convert extended UD fore to simple UD format
    
    Parameters:
    read_path (str): path to the directory with extended UD files.
    save_path (str): path to save the converted corpus.  
    '''
    print("Converting extended UD to base UD...")
    
    # create save directory if non existent
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    
    for file_path in pathlib.Path(read_path).rglob("**/*.conllu"):
        conll = pyconll.load_from_file(file_path)
        
        for sent_id, sent in enumerate(conll, start=1):
            
            ellipsis = 0 
            id_map = {'0': '0'}
            dependents = {}
            for new_id, token in enumerate(sent, start=1):
                if token.deps and len(token.deps) == 1:
                    for head, dep in token.deps.items():
                        deprel = dep[0]
                        if '.' in head:
                            if head not in dependents:
                                dependents[head] = []
                            dependents[head].append((token, deprel))
                            
                if token.is_empty_node():
                    ellipsis += 1
                    if token.id not in dependents:
                        dependents[token.id] = []
                    
                id_map[token.id] = new_id
            
            
            if ellipsis:
                restore_ellipsis(sent, dependents)             
                
                for new_id, token in enumerate(sent, start=1):
                    token.head = str(id_map[token.head])
                    
                    token.id = str(new_id)
                    
            for token in sent:
                token.deps = {}
            
            source, id = get_ud_source(sent)
            sent.set_meta('sent_id', f"{id}_{source}")
        
        # save the converted corpus
        save_file_path = pathlib.Path(save_path) / file_path.name
        with open(save_file_path, 'w', encoding='utf-8') as f:
            conll.write(f)
    
    print("Conversion completed.")

            
            
def convert_str_to_ud(read_path, save_path):
    """
    Конвертация SynTagRus -> UD c корректным назначением head с учётом пунктуации.
    Ключевая идея: DOM в STR = ID головы (W/@ID), а не порядковая позиция.
    Поэтому делаем 2 прохода:
      1) собираем ленту (punct/word) и считаем итоговые token_id для ВСЕХ слов,
         строим карту: word_ID -> token_id;
      2) генерим строки; head для слова берём по карте из его DOM-ID.
    Пунктуация вставляется как отдельные токены и не участвует в head слов.
    """
    import pathlib
    import xml.etree.ElementTree as ET
    from convert.utils import get_str_source

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

    def _word_upos(w_el) -> str:
        feat = (w_el.get('FEAT') or '').strip()
        return feat if feat else '_'

    def _word_id(w_el) -> str:
        wid = (w_el.get('ID') or '').strip()
        return wid if wid else str(id(w_el))  # запасной уникальный id

    def _word_dom_id(w_el):
        """Вернуть строковый ID головы из DOM, если он задан (не '_root'/'0'), иначе None."""
        dom = (w_el.get('DOM') or '').strip()
        if not dom or dom == '_root' or dom == '0':
            return None
        return dom  # это именно ID узла-головы

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

                # 1) Собираем ленту: ('punct', ch) | ('word', w_el)
                seq = []

                # пунктуация ДО первого слова
                for ch in _iter_visible_punct((sent.text or '').strip()):
                    seq.append(('punct', ch))

                # слова + пунктуация ПОСЛЕ каждого слова
                for w in wlist:
                    seq.append(('word', w))
                    for ch in _iter_visible_punct((w.tail or '').strip()):
                        seq.append(('punct', ch))

                # если последнее слово содержит приклеенную точку — отделим её
                if seq and seq[-1][0] == 'word':
                    last_w = seq[-1][1]
                    lf = (last_w.text or '')
                    if lf.endswith('.'):
                        last_w.text = lf[:-1] if lf[:-1] else '_'
                        seq.append(('punct', '.'))

                # 2) ПЕРВЫЙ проход: считаем окончательные token_id слов и строим карту W/@ID -> token_id
                wid_to_tokenid: dict[str, str] = {}
                token_id = 0
                for kind, obj in seq:
                    token_id += 1
                    if kind == 'word':
                        wid_to_tokenid[_word_id(obj)] = str(token_id)

                # 3) ВТОРОЙ проход: формируем строки CoNLL-U
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
                        upos = _word_upos(w_el)
                        dom_id = _word_dom_id(w_el)
                        deprel = _word_deprel(w_el) or '_'

                        if dom_id is None:
                            head = '0'
                            deprel_out = 'root'
                        else:
                            head = wid_to_tokenid.get(dom_id, '0')
                            deprel_out = deprel
                            # защита от самоссылки (может случиться при кривых DOM/ID)
                            if head == str(token_id):
                                head = '0'
                                deprel_out = 'root'

                        parts = [
                            str(token_id),  # id
                            form,           # form
                            '_',            # lemma
                            upos,           # upos (оставляем str-тег)
                            '_',            # xpos
                            '_',            # feats
                            head,           # head
                            deprel_out,     # deprel
                            '_',            # deps
                            '_'             # misc
                        ]
                        lines.append('\t'.join(parts))

                # 4) мета и сборка
                text_line = ' '.join(p.split('\t')[1] for p in lines if p.split('\t')[1] != '_')
                header = [
                    f"# sent_id = {sent_idx}_{source}",
                    f"# text = {text_line}",
                ]
                conll = '\n'.join(header + lines) + '\n'
                conll_chunks.append(conll)

        # 5) сохранить
        rel = ''.join(file_path.parts[1:])
        rel = rel[: rel.rfind('.tgt')]
        out_path = out_root / (rel + '.conllu')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(conll_chunks))

    print("Conversion completed.")
