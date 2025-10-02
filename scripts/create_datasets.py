from pathlib import Path
import random

SEED = 42
TRAIN_RATIO, DEV_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

def read_blocks(path: Path):
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    return [b for b in text.split("\n\n") if b.strip()]

def write_blocks(path: Path, blocks):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

def main():
    str_path = Path("Aligned/str_aligned.conllu")
    ud_path  = Path("Aligned/ud_aligned.conllu")

    str_blocks = read_blocks(str_path)
    ud_blocks  = read_blocks(ud_path)

    if len(str_blocks) != len(ud_blocks):
        raise ValueError(f"Несовпадение количества предложений: STR={len(str_blocks)} UD={len(ud_blocks)}")

    n = len(str_blocks)
    idx = list(range(n))
    random.Random(SEED).shuffle(idx)

    n_train = int(n * TRAIN_RATIO)
    n_dev   = int(n * DEV_RATIO)
    n_test  = n - n_train - n_dev

    splits = {
        "train": idx[:n_train],
        "dev":   idx[n_train:n_train+n_dev],
        "test":  idx[n_train+n_dev:],
    }

    out_root = Path("datasets")
    for split, ids in splits.items():
        write_blocks(out_root / "str" / f"{split}.conllu", [str_blocks[i] for i in ids])
        write_blocks(out_root / "ud"  / f"{split}.conllu", [ud_blocks[i]  for i in ids])

    print(f"Готово. Всего {n} предложений → train={len(splits['train'])}, dev={len(splits['dev'])}, test={len(splits['test'])}")
    print(f"Файлы сохранены в {out_root}/...")

if __name__ == "__main__":
    main()
