# convert/promote_ne4.py
from typing import List, Dict, Set, Optional
from pyconll.unit.sentence import Sentence
from pyconll.unit.token import Token

# Полный список НЕ4 форм и их «корней» (вторых слов)
_NE4_FORMS: Set[str] = {
    "незачем", "нечего", "некуда", "неоткуда", "некого", "некогда",
    "непочем", "нечем", "негде", "некому", "нечему"
}
_NE4_BODIES: Set[str] = {
    "зачем", "чего", "куда", "откуда", "кого", "когда",
    "почем", "чем", "где", "кому", "чему"
}

_PUNCT_CHARS = '''"'“”«»„‚()[]{}.,;:!? \t\r\n'''


def _norm_form(s: str) -> str:
    if not s:
        return ""
    s = s.lower().replace("ё", "е").strip(_PUNCT_CHARS)
    return s


def _rows_from_sentence(sent: Sentence) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for t in sent:
        head = t.head if t.head not in (None, "", "_") else "0"
        is_punct = (t.upos or "") == "PUNCT"
        if t.deprel:
            deprel = t.deprel
        else:
            deprel = "root" if (head == "0" and not is_punct) else "_"

        rows.append({
            "id": str(t.id),
            "form": (t.form if t.form not in (None, "") else "_"),
            "lemma": "_",
            "upos": (t.upos or "_"),
            "xpos": "_",
            "feats": "_",
            "head": str(head),
            "deprel": deprel,
            "is_punct": "1" if is_punct else "0",
        })
    return rows


def _children_map(rows: List[Dict[str, str]]) -> Dict[str, List[int]]:
    deps: Dict[str, List[int]] = {}
    for idx, r in enumerate(rows):
        h = r["head"]
        if h != "0":
            deps.setdefault(h, []).append(idx)
    return deps


def _build_token_line(id_: str, r: Dict[str, str]) -> str:
    return (
        f"{id_}\t{r['form']}\t{r['lemma']}\t{r['upos']}\t{r['xpos']}\t"
        f"{r['feats']}\t{r['head']}\t{r['deprel']}\t_\t_"
    )


def _reindex_rows(rows: List[Dict[str, str]], src_sent: Sentence) -> Sentence:
    old2new: Dict[str, str] = {}
    for new_i, r in enumerate(rows, start=1):
        old2new[r["id"]] = str(new_i)

    for r in rows:
        r["id"] = old2new[r["id"]]
        h = r["head"]
        r["head"] = (old2new.get(h, "0") if h != "0" else "0")
        if r["head"] == "0":
            if r["is_punct"] == "1":
                if not r["deprel"] or r["deprel"].lower() == "root":
                    r["deprel"] = "_"
            else:
                r["deprel"] = "root"
        else:
            if (r["deprel"] or "").lower() == "root":
                r["deprel"] = "dep"

    sid = src_sent.meta_value("sent_id") or ""
    text = " ".join([r["form"] for r in rows if r["form"] not in ("_", "")])
    meta = []
    if sid:
        meta.append(f"# sent_id = {sid}")
    meta.append(f"# text = {text}")
    body = "\n".join(_build_token_line(r["id"], r) for r in rows)
    return Sentence("\n".join(meta + [body]))


def _is_punct(tok: Token) -> bool:
    ch = (tok.form or '').lower()
    return tok.upos == "PUNCT" and not ch in ('x', '×')


def _is_ellipsis_form(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    return s == "_" or s.lower() == "<x>"


def _ud_has_ellipsis(ud_sent: Optional[Sentence]) -> bool:
    """
    Определяем, есть ли в UD-версии явный эллипсис вида <X>.
    Держим проверку максимально простой и безопасной.
    """
    if ud_sent is None:
        return False
    for t in ud_sent:
        if (t.form is None or t.form == "_") and not _is_punct(t):
            return True
    return False


def _str_has_ellipsis(rows: List[Dict[str, str]]) -> bool:
    # Детектируем эллипсис в STR по форме "_" или "<X>"
    return any(_is_ellipsis_form(r["form"]) for r in rows)

def _is_punct_token(upos: Optional[str]) -> bool:
    return (upos or "") == "PUNCT"

def _ud_has_split_ne4(ud_sent: Optional[Sentence]) -> bool:
    """
    Проверяем, есть ли в UD последовательность: НЕ + (зачем/куда/когда/чем/...)
    (между ними может быть пунктуация). Если есть, то в STR склейку делать не надо.
    """
    if ud_sent is None:
        return False
    ud_tokens = list(ud_sent)
    m = len(ud_tokens)
    i = 0
    while i < m:
        ti = ud_tokens[i]
        if _norm_form(ti.form or "") == "не":
            j = i + 1
            # пропустим пунктуацию между "не" и словом-основанием
            while j < m and _is_punct_token(ud_tokens[j].upos):
                j += 1
            if j < m:
                tj = ud_tokens[j]
                body = _norm_form(tj.form or "")
                if body in _NE4_BODIES:
                    return True
        i += 1
    return False


def promote_ne4_constructions_in_str(
    str_sent: Sentence,
    ud_sent: Optional[Sentence] = None,
) -> Sentence:
    """
    Склейка НЕ4-конструкций в STR.

    Защиты:
    1) Если в UD есть <X> и в STR присутствует эллипсис ("_" или "<X>"),
       то НЕ трогаем эллипсис в STR (пропускаем PASS A).
    2) Если в UD есть раздельное написание НЕ + (зачем/куда/когда/чем/...),
       то НЕ выполняем склейку в STR (пропускаем PASS B).
    """

    rows = _rows_from_sentence(str_sent)
    if not rows:
        return str_sent

    deps = _children_map(rows)
    to_delete = set()
    n = len(rows)

    protect_ellipsis = _ud_has_ellipsis(ud_sent) and _str_has_ellipsis(rows)
    protect_split_ne4 = _ud_has_split_ne4(ud_sent)

    # ---------- PASS A: эллипсис "_" или "<X>" + [НЕ4] ----------
    # Если эллипсис в UD подтверждён и в STR уже есть эллипсис, не трогаем его.
    if not protect_ellipsis:
        # индексы эллипсиса
        ell_idxs = [i for i, r in enumerate(rows) if _is_ellipsis_form(r["form"])]
        # ищем НЕ4 только СПРАВА (не смотрим влево)
        offsets = (1, 2, 3)

        for i in ell_idxs:
            word_j = None
            for d in offsets:
                j = i + d
                if 0 <= j < n and j != i:
                    rj = rows[j]
                    if rj["is_punct"] == "1":
                        continue
                    if _norm_form(rj["form"]) in _NE4_FORMS and not _is_ellipsis_form(rj["form"]):
                        word_j = j
                        break
            if word_j is None:
                continue

            # первый = эллипсис, второй = слитная НЕ4-форма (некуда/некогда/…)
            first = rows[i]
            second = rows[word_j]
            first_id = first["id"]
            second_id = second["id"]

            # (1) первый токен получает словоформу второго
            first["form"] = second["form"]

            # (2) head/deprel:
            #     по умолчанию сохраняем свои; если первый зависим от второго (head == second_id),
            #     то наоборот — берём head/deprel второго (с защитой от самоссылки).
            if first["head"] == second_id:
                new_head = second["head"]
                new_deprel = second.get("deprel", first.get("deprel", "_"))
                # защита от самоссылки
                if new_head == first_id:
                    new_head = "0"
                    # если оказались корнем — корректно проставим метку
                    if (new_deprel or "").lower() != "root":
                        new_deprel = "root"
                first["head"] = new_head
                first["deprel"] = new_deprel
            # иначе head/deprel остаются как были у first
            
            # (3) первый токен получает морфологию второго
            first["upos"] = second.get("upos", first["upos"])
            first["xpos"] = second.get("xpos", first.get("xpos", "_"))
            first["feats"] = second.get("feats", first.get("feats", "_"))

            # (4) все дети второго токена привязываются к первому
            for ch_idx in deps.get(second_id, []):
                # не переназначаем самого first (на случай странных дуг)
                if rows[ch_idx]["id"] != first_id:
                    rows[ch_idx]["head"] = first_id
                    if (rows[ch_idx]["deprel"] or "").lower() == "root":
                        rows[ch_idx]["deprel"] = "dep"

            # удалить второй токен
            to_delete.add(word_j)

        if to_delete:
            rows = [r for idx, r in enumerate(rows) if idx not in to_delete]
            deps = _children_map(rows)
            to_delete.clear()
            n = len(rows)

    # ---------- PASS B: "не" + (зачем/куда/когда/чем/...) -> НЕ4 ----------
    if not protect_split_ne4:
        i = 0
        while i < n - 1:
            r1 = rows[i]
            r2 = rows[i + 1]
            if r1["is_punct"] == "1" or r2["is_punct"] == "1":
                i += 1
                continue

            if _norm_form(r1["form"]) == "не" and not _is_ellipsis_form(r2["form"]):
                body = _norm_form(r2["form"])
                target_form = "не" + body
                if body in _NE4_BODIES and target_form in _NE4_FORMS:
                    first = r1          # "не"
                    second = r2         # {куда, когда, чем, ...}
                    first_id = first["id"]
                    second_id = second["id"]

                    # (1) первый токен получает словоформу второго ИЗ СКЛЕЙКИ (слитная НЕ4-форма)
                    first["form"] = target_form

                    # (2) head/deprel:
                    #     по умолчанию сохраняем свои; если первый зависим от второго (head == second_id),
                    #     то наоборот — берём head/deprel второго (с защитой от самоссылки).
                    if first["head"] == second_id:
                        new_head = second["head"]
                        new_deprel = second.get("deprel", first.get("deprel", "_"))
                        if new_head == first_id:
                            new_head = "0"
                            if (new_deprel or "").lower() != "root":
                                new_deprel = "root"
                        first["head"] = new_head
                        first["deprel"] = new_deprel
                    # иначе head/deprel остаются как были у first

                    # (3) первый токен получает морфологию второго
                    first["upos"] = second.get("upos", first["upos"])
                    first["xpos"] = second.get("xpos", first.get("xpos", "_"))
                    first["feats"] = second.get("feats", first.get("feats", "_"))

                    # (4) все дети второго токена привязываются к первому
                    for ch_idx in deps.get(second_id, []):
                        if rows[ch_idx]["id"] != first_id:
                            rows[ch_idx]["head"] = first_id
                            if (rows[ch_idx]["deprel"] or "").lower() == "root":
                                rows[ch_idx]["deprel"] = "dep"

                    # удалить второй токен
                    to_delete.add(i + 1)

                    # после удаления длина уменьшится — пересчитаем n и не перескакиваем позицию
                    i += 1
                    continue

            i += 1

    if to_delete:
        rows = [r for idx, r in enumerate(rows) if idx not in to_delete]

    return _reindex_rows(rows, str_sent)
