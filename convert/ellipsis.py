from typing import List, Dict, Set
from pyconll.unit.sentence import Sentence
from pyconll.unit.token import Token


def reconcile_ellipsis_by_syntax(ud_sent: Sentence, str_sent: Sentence) -> Sentence:
    # ---------- утилиты (в стиле вашего кода) ----------
    def _is_punct(tok: Token) -> bool:
        return (tok.upos or "") == "PUNCT"

    def _is_ellipsis(tok: Token) -> bool:
        return (tok.form is None or tok.form == "_") and not _is_punct(tok)

    def _norm(s) -> str:
        return (s or "").lower().replace("ё", "е")

    def _build_dependents(tokens: List[Token]) -> Dict[str, List[Token]]:
        deps: Dict[str, List[Token]] = {}
        for t in tokens:
            h = str(t.head) if t.head not in (None, "_") else "0"
            deps.setdefault(h, []).append(t)
        return deps

    def _runs(idxs: List[int], tokens: List[Token]) -> List[List[int]]:
        # объединяем эллипсисы в «забеги», игнорируя промежуточную пунктуацию
        if not idxs:
            return []
        idxs = sorted(idxs)
        out: List[List[int]] = []
        cur = [idxs[0]]
        for a, b in zip(idxs, idxs[1:]):
            only_punct = all(_is_punct(tokens[k]) for k in range(a + 1, b))
            if only_punct:
                cur.append(b)
            else:
                out.append(cur)
                cur = [b]
        if cur:
            out.append(cur)
        return out

    def _external_out_head(tok: Token, run_ids: Set[str]) -> str:
        h = str(tok.head) if tok.head not in (None, "_") else "0"
        return h if (h != "0" and h not in run_ids) else "0"

    def _choose_new_root_child_using_ud(
        ud_sent: Sentence,
        tok: Token,
        run_ids: Set[str],
        deps: Dict[str, List[Token]],
    ):
        """ среди внешних детей удаляемого корня выбрать нового (предпочитая не-PUNCT) """
        children = deps.get(str(tok.id), []) or []
        ext_children = [ch for ch in children if str(ch.id) not in run_ids]
        if not ext_children:
            return None

        # попробуем совместить по форме с UD-root
        ud_root = None
        for u in ud_sent:
            if str(u.head) == "0" or ((u.deprel or "").lower() == "root"):
                if (u.upos or "") != "PUNCT":
                    ud_root = u
                    break
                ud_root = u
                break
        if ud_root is not None:
            target = _norm(ud_root.form)
            for ch in ext_children:
                if not _is_punct(ch) and _norm(ch.form) == target:
                    return ch

        for ch in ext_children:
            if not _is_punct(ch):
                return ch
        return ext_children[0]

    def _reindex(tokens_new: List[Token], orig_sent: Sentence) -> Sentence:
        # карта старый id -> новый id
        old2new: Dict[str, str] = {}
        for new_i, t in enumerate(tokens_new, start=1):
            old2new[str(t.id)] = str(new_i)
        # перенумерация и маппинг голов
        for new_i, t in enumerate(tokens_new, start=1):
            t.id = str(new_i)
            h = str(t.head) if t.head not in (None, "_") else "0"
            t.head = old2new.get(h, "0") if h != "0" else "0"
        # собрать conllu
        sid = orig_sent.meta_value("sent_id") or ""
        text = " ".join([t.form for t in tokens_new if t.form and t.form != "_"])
        meta = []
        if sid:
            meta.append(f"# sent_id = {sid}")
        meta.append(f"# text = {text}")
        from pyconll import load_from_string
        body = "\n".join(
            "\t".join(
                [
                    t.id,
                    t.form or "_",
                    t.lemma or "_",
                    t.upos or "_",
                    t.xpos or "_",
                    t.feats or "_",
                    (str(t.head) if t.head not in (None, "_") else "0"),
                    t.deprel or "_",
                    t.deps or "_",
                    t.misc or "_",
                ]
            )
            for t in tokens_new
        )
        ds = load_from_string("\n".join(meta + [body]) + "\n")
        for s in ds:
            return s
        return orig_sent

    # ---------- основной код ----------
    _root_ellipsis_deleted = False
    _str_before_conll = str_sent.conll()
    
    tokens: List[Token] = list(str_sent)
    if not tokens:
        return str_sent

    deps = _build_dependents(tokens)

    # индексы эллипсисов
    ell_idxs_str = [i for i, t in enumerate(tokens) if _is_ellipsis(t)]
    if not ell_idxs_str:
        return str_sent
    ell_idxs_ud = [i for i, t in enumerate(ud_sent) if _is_ellipsis(t)]

    str_runs = _runs(ell_idxs_str, tokens)
    ud_runs = _runs(ell_idxs_ud, list(ud_sent))

    desired_sizes = [len(ud_runs[k]) if k < len(ud_runs) else 0 for k in range(len(str_runs))]

    to_delete: Set[int] = set()

    # обработка каждой группы эллипсиса
    for k, run in enumerate(str_runs):
        n = len(run)
        m_desired = desired_sizes[k]
        if n == m_desired:
            continue

        deps = _build_dependents(tokens)
        run_tokens: List[Token] = [tokens[i] for i in run]
        run_ids: Set[str] = {str(t.id) for t in run_tokens}

        # полностью удалить группу
        if m_desired == 0:
            for idx in run:
                t = tokens[idx]
                is_root = (str(t.head) == "0") or ((t.deprel or "").lower() == "root")
                if is_root:
                    _root_ellipsis_deleted = True
                    new_root = _choose_new_root_child_using_ud(ud_sent, t, run_ids, deps)
                    if new_root is not None:
                        new_root.head = "0"
                        new_root.deprel = "root"
                        # внешние + внутренние (если вдруг такие есть) дети → к new_root
                        for ch in deps.get(str(t.id), []) or []:
                            if ch is not new_root:
                                ch.head = str(new_root.id)
                                if (ch.deprel or "").lower() == "root":
                                    ch.deprel = "dep"
                    else:
                        # всех детей к '0'
                        for ch in deps.get(str(t.id), []) or []:
                            ch.head = "0"
                else:
                    tgt = _external_out_head(t, run_ids)
                    for ch in deps.get(str(t.id), []) or []:
                        if str(ch.id) not in run_ids:
                            ch.head = tgt
                        else:
                            # внутренний сохраняемый ребёнок (другой эллипсис) → к внешней голове удаляемого, либо к tgt
                            parent_h = _external_out_head(t, run_ids)
                            ch.head = parent_h if parent_h != "0" else tgt
                            if (ch.deprel or "").lower() == "root" and ch.head != "0":
                                ch.deprel = "dep"

                to_delete.add(idx)
            continue

        # частичное удаление внутри группы
        # профили (сохраняем ваш способ сортировки)
        def incoming_count(tok_id: str) -> int:
            return sum(1 for d in deps.get(tok_id, []) if str(d.id) not in run_ids and not _is_ellipsis(d))

        def has_outgoing(tok: Token) -> bool:
            return _external_out_head(tok, run_ids) != "0"

        profiles = []
        mid = (n - 1) / 2.0
        for pos_in_run, t in enumerate(run_tokens):
            profiles.append(
                {
                    "tok": t,
                    "idx": run[pos_in_run],
                    "incoming": incoming_count(str(t.id)),
                    "outgoing": has_outgoing(t),
                    "center": -(abs(pos_in_run - mid)),
                }
            )

        profiles_sorted = sorted(
            profiles, key=lambda p: (p["incoming"], 1 if p["outgoing"] else 0, p["center"])  # type: ignore
        )
        keep: Set[str] = set(str(p["tok"].id) for p in profiles_sorted[-m_desired:])  # type: ignore

        # primary — лучший из keep с исходящей, иначе последний из keep
        primary_id = next(
            (str(p["tok"].id) for p in reversed(profiles_sorted) if str(p["tok"].id) in keep and p["outgoing"]),  # type: ignore
            None,
        )
        if primary_id is None:
            primary_id = next(iter(keep))

        # удаляем всё, что не в keep
        for p in profiles_sorted:
            tid = str(p["tok"].id)  # type: ignore
            if tid in keep:
                continue

            is_root = (str(p["tok"].head) == "0") or ((p["tok"].deprel or "").lower() == "root")  # type: ignore
            all_children = (deps.get(tid, []) or [])
            ext_children = [ch for ch in all_children if str(ch.id) not in run_ids]
            int_kept_children = [ch for ch in all_children if str(ch.id) in keep]  # <-- НОВОЕ: внутренние сохраняемые

            if is_root:
                # новый корень — внешний ребёнок (или fallback)
                _root_ellipsis_deleted = True
                new_root = _choose_new_root_child_using_ud(ud_sent, p["tok"], run_ids, deps)  # type: ignore
                if new_root is not None:
                    new_root.head = "0"
                    new_root.deprel = "root"
                    # внешние дети → к new_root
                    for ch in ext_children:
                        if ch is not new_root:
                            ch.head = str(new_root.id)
                            if (ch.deprel or "").lower() == "root":
                                ch.deprel = "dep"
                    # ВНУТРЕННИЕ СОХРАНЯЕМЫЕ дети → к new_root
                    for ch in int_kept_children:
                        ch.head = str(new_root.id)
                        if (ch.deprel or "").lower() == "root":
                            ch.deprel = "dep"
                else:
                    # fallback: все дети → к '0'
                    for ch in ext_children + int_kept_children:
                        ch.head = "0"
                        if (ch.deprel or "").lower() == "root" and ch.head != "0":
                            ch.deprel = "dep"
            else:
                # Некорневой удаляемый:
                # 1) внешние дети → к primary
                for ch in ext_children:
                    ch.head = primary_id
                    if (ch.deprel or "").lower() == "root":
                        ch.deprel = "dep"
                # 2) ВНУТРЕННИЕ СОХРАНЯЕМЫЕ дети:
                #    предпочитаем голову удаляемого, если она остаётся; иначе внешнюю голову удаляемого; иначе primary
                parent_h = str(p["tok"].head) if p["tok"].head not in (None, "_") else "0"  # type: ignore
                head_candidate = None
                if parent_h in keep:
                    head_candidate = parent_h
                elif parent_h not in run_ids and parent_h != "0":
                    head_candidate = parent_h
                else:
                    head_candidate = primary_id

                for ch in int_kept_children:
                    ch.head = head_candidate
                    if (ch.deprel or "").lower() == "root" and ch.head != "0":
                        ch.deprel = "dep"

            to_delete.add(p["idx"])  # type: ignore

    if not to_delete:
        return str_sent

    # ---------- ПОСТ-ПРОХОД: починка висячих голов ----------
    # Любой оставшийся токен, чей head указывает на удалённый id, привяжем к первому «живому» предку
    survivors: Set[str] = {str(t.id) for i, t in enumerate(tokens) if i not in to_delete}
    id2head: Dict[str, str] = {str(t.id): (str(t.head) if t.head not in (None, "_") else "0") for t in tokens}

    def _first_surviving_ancestor(hid: str) -> str:
        seen: Set[str] = set()
        cur = hid
        while cur != "0" and cur not in survivors and cur not in seen:
            seen.add(cur)
            cur = id2head.get(cur, "0")
        return cur if (cur in survivors or cur == "0") else "0"

    for i, t in enumerate(tokens):
        if i in to_delete:
            continue
        h = str(t.head) if t.head not in (None, "_") else "0"
        if h != "0" and h not in survivors:
            new_h = _first_surviving_ancestor(h)
            # защита от самоссылки
            if new_h == str(t.id):
                new_h = "0"
            t.head = new_h
            if (t.deprel or "").lower() == "root" and new_h != "0":
                t.deprel = "dep"

    # собрать результат
    kept_tokens: List[Token] = [t for i, t in enumerate(tokens) if i not in to_delete]
    for t in kept_tokens:
        if t.upos != "PUNCT":
            t.misc = {}
            
    res = _reindex(kept_tokens, str_sent)
    
    try:
        if _root_ellipsis_deleted:
            with open('root_ellipsis.log', 'a', encoding='utf-8') as lf:
                lf.write('UD:\n')
                lf.write(ud_sent.conll() + '\n')
                lf.write('\nSynTagRus_before:\n')
                lf.write(_str_before_conll + '\n')
                lf.write('\nSynTagRus_after:\n')
                lf.write(res.conll() + '\n')
                lf.write('-' * 100 + '\n')
    except Exception:
        pass
    
    return res
