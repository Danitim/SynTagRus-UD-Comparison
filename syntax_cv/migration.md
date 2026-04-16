# Миграция со старого API на новый

Краткая сводка изменений, которые ломают совместимость с прежним
`align_markup.py` / `lca_analysis.py`.

## Что удалено

| Старая функция | Причина удаления |
|---|---|
| `swap_rows_in_df` | Меняет STR-строки местами; запрещено куратором. |
| `apply_all_swaps` | То же; fa пуассон на intersection пары производил синтетические аналитические связи. |
| `apply_swap_plan` | Композиция удалённых функций. |
| `build_recursive_swap_plan(_cp)` | «Рекурсия» имела смысл только при мутации дерева; в неизменяемом сеттинге CP стабилизируется в один проход. |
| `apply_token_mapping` | Переиндексация строк — ещё один вид мутации. |
| `build_token_mapping_from_plan` | Выход swap-плана. |
| `find_pairs_to_swap`, `find_pairs_to_swap_cp` | Устарели как самостоятельные функции; их логика поглощена `build_edge_correspondence`. |
| `get_unresolved_alignment_sentences`, `inspect_unresolved_alignment_sentence` | Заменены `diagnose_sentences` и `SentenceCorrespondence.diagnostics`. |

## Что мигрировано

| Было | Стало |
|---|---|
| `align_markup.resolve_edge_matching` | `resolve.resolve_edge_matching` (с новым флагом `allow_open_fallback=False`) |
| `align_markup.classify_edges` | оставлено как есть в виде read-only утилиты, но канонический путь — через `EdgeCorrespondence` |
| `align_markup.compare_matched_edges` | `token_comparison.build_comparison_table(..., status=['exact_same_dir'])` |
| `lca_analysis.build_lca_triples` | полностью сохраняется как отдельный модуль; изолирован от нового пайплайна и используется только для аналитических отчётов |
| `utils.build_data(aligned=True)` | `build_data(...)` + отдельный вызов `build_correspondence(...)` |

## Новый публичный API (имена импортировать отсюда)

```python
from edge_correspondence import (
    EdgeCorrespondence,
    SentenceCorrespondence,
    EdgeMatch,
    build_edge_correspondence,
)
from resolve import resolve_edge_matching
from lca_candidates import build_lca_candidates
from token_comparison import (
    build_comparison_table,
    label_confusion,
    coverage_by_mode,
)
from examples_finder import (
    diagnose_sentences,
    pick_examples,
    category_summary,
)
from tikz_dep import (
    render_pair,
    render_document,
    render_category_album,
)
from utils import (
    build_data,
    filter_consistent,
    build_correspondence,
    attach_comparisons,
)
```

## Как переписать типичные места в коде

### 1. Раньше: `align_recursively` перед метриками

```python
# ДО
str_aligned = align_recursively(str_df, ud_df, method="cp")
# compare_df = compare on str_aligned vs ud_df
```

```python
# ПОСЛЕ
corr = build_edge_correspondence(str_df, ud_df, mode="strict")
compare_df = build_comparison_table(str_df, ud_df, corr)
comparable = compare_df[compare_df["comparable"]]
```

### 2. Раньше: расчёт покрытия через число swap-пар

```python
# ДО
plan = build_recursive_swap_plan_cp(str_df, ud_df)
n_pairs = sum(len(p) for p in plan)
# никакой явной метрики покрытия не было
```

```python
# ПОСЛЕ
corr = build_edge_correspondence(str_df, ud_df, mode="extended",
                                 extra_candidates=build_lca_candidates(str_df, ud_df))
stats = corr.coverage()
# {'resolved_pct': 95.1, 'baseline_pct': 58.3, ...}
```

### 3. Раньше: отчёт о неразрешённых случаях

```python
# ДО
df = get_unresolved_alignment_sentences(str_df, ud_df)
```

```python
# ПОСЛЕ
diag = diagnose_sentences(str_df, ud_df)
bad = [s for s, d in diag.items() if d.category == 5]
summary = category_summary(diag)
```

### 4. Раньше: ручные картинки через tikz

```python
# ПОСЛЕ: авто-генерация для 5 категорий куратора
picks = pick_examples(diag, per_category=1, ud_df=ud_df)
corr_strict = build_edge_correspondence(str_df, ud_df, mode="strict")
corr_ext = build_edge_correspondence(
    str_df, ud_df, mode="extended",
    extra_candidates=build_lca_candidates(str_df, ud_df),
)
album = render_category_album(str_df, ud_df, corr_strict, corr_ext, picks)
for cat, sid, tex in album:
    Path(f"example_cat{cat}_{sid}.tex").write_text(tex, encoding="utf-8")
```

## Соответствие понятий куратора и новых статусов

| Куратор | Новый статус |
|---|---|
| «Структуры совпадают» | `exact_same_dir` |
| «Можно поменять слова местами» | `exact_mirrored` |
| «Нужен LCA-путь» | `restructured` (доступно только в `mode="extended"`) |
| «Метод не сработал» | `unresolved` |
| «Корректно сопоставляется» | `comparable == True` |

## Инварианты, которые полезно проверять автотестами

1. `C_strict ⊆ C_extended`: для любого предложения множество
   разрешённых в strict edges содержится в extended. Проверка:
   `all(sid not in bad_cat4 for sid, d in diag.items())` —
   категория 4 должна быть пуста.
2. Injectivity: $\varphi$ инъективна на фиксированных парах —
   непосредственно следует из `used_str.add(...)` в `resolve.py`.
3. Идемпотентность: повторный вызов `build_edge_correspondence` с
   теми же входами даёт тот же результат бит-в-бит (нет зависимости
   от порядка итерации pandas).