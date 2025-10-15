import argparse
from pathlib import Path
import random
from typing import List, Tuple, Dict

NAME_TO_SUBDIR: Dict[str, str] = {
    "ud_aligned.conllu":  "ud",
    "str_aligned.conllu": "str",
    "ud_new.conllu":      "ud_new",
    "str_new.conllu":     "str_new",
    "ud_old.conllu":      "ud_old",
    "str_old.conllu":     "str_old",
}

PAIRS = [
    ("ud_aligned.conllu", "str_aligned.conllu"),
    ("ud_new.conllu",     "str_new.conllu"),
    ("ud_old.conllu",     "str_old.conllu"),
]

def read_sentences(path: Path) -> List[List[str]]:
    sents: List[List[str]] = []
    cur: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                if cur:
                    sents.append(cur)
                    cur = []
            else:
                cur.append(line)
        if cur:
            sents.append(cur)
    return sents

def write_sentences(path: Path, sents: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        for i, sent in enumerate(sents):
            for ln in sent:
                f.write(ln if ln.endswith("\n") else ln + "\n")
            if i != len(sents) - 1:
                f.write("\n")

def calc_split_indices(n: int, train_r: float, dev_r: float, test_r: float,
                       shuffle: bool, seed: int) -> Tuple[List[int], List[int], List[int]]:
    assert n >= 0
    idxs = list(range(n))
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(idxs)

    n_train = int(n * train_r)
    n_dev   = int(n * dev_r)
    n_test  = int(n * test_r)

    used = n_train + n_dev + n_test
    rem = n - used
    for bucket in ("train", "dev", "test"):
        if rem <= 0:
            break
        if bucket == "train":
            n_train += 1
        elif bucket == "dev":
            n_dev += 1
        else:
            n_test += 1
        rem -= 1

    train_idx = idxs[:n_train]
    dev_idx   = idxs[n_train:n_train + n_dev]
    test_idx  = idxs[n_train + n_dev:n_train + n_dev + n_test]
    return train_idx, dev_idx, test_idx

def split_one_file(src: Path, dst_root: Path,
                   train_idx: List[int], dev_idx: List[int], test_idx: List[int]) -> Tuple[int, int, int]:
    sents = read_sentences(src)
    total = len(sents)
    def pick(ixs: List[int]) -> List[List[str]]:
        return [sents[i] for i in ixs]

    subdir = NAME_TO_SUBDIR[src.name]
    write_sentences(dst_root / subdir / "train" / "data.conllu", pick(train_idx))
    write_sentences(dst_root / subdir / "dev"   / "data.conllu", pick(dev_idx))
    write_sentences(dst_root / subdir / "test"  / "data.conllu", pick(test_idx))
    return total, len(train_idx), len(dev_idx), len(test_idx)

def process_pair(a: Path, b: Path, dst_root: Path,
                 train_r: float, dev_r: float, test_r: float,
                 shuffle: bool, seed: int) -> None:
    sents_a = read_sentences(a)
    sents_b = read_sentences(b)
    if len(sents_a) != len(sents_b):
        msg = f"[WARN] mismatch sizes for pair {a.name}<->{b.name}: {len(sents_a)} vs {len(sents_b)}"
        raise SystemExit(msg)

    n = len(sents_a)
    train_idx, dev_idx, test_idx = calc_split_indices(n, train_r, dev_r, test_r, shuffle, seed)

    def write_from_list(src_name: str, sents: List[List[str]]):
        subdir = NAME_TO_SUBDIR[src_name]
        write_sentences(dst_root / subdir / "train.conllu", [sents[i] for i in train_idx])
        write_sentences(dst_root / subdir / "dev.conllu", [sents[i] for i in dev_idx])
        write_sentences(dst_root / subdir / "test.conllu", [sents[i] for i in test_idx])

    write_from_list(a.name, sents_a)
    write_from_list(b.name, sents_b)

    print(f"[OK] pair {a.name} & {b.name}: total={n}, train={len(train_idx)}, dev={len(dev_idx)}, test={len(test_idx)}")

def main():
    ap = argparse.ArgumentParser(description="Разбивает conllu из Aligned/ на train/dev/test с сохранением выравнивания UD↔STR.")
    ap.add_argument("--src", default="Aligned", help="Каталог c исходными .conllu (по умолчанию: Aligned)")
    ap.add_argument("--dst", default="datasets", help="Каталог назначения (по умолчанию: datasets)")
    ap.add_argument("--train-ratio", type=float, default=0.8, help="Доля train (по умолчанию 0.8)")
    ap.add_argument("--dev-ratio",   type=float, default=0.1, help="Доля dev (по умолчанию 0.1)")
    ap.add_argument("--test-ratio",  type=float, default=0.1, help="Доля test (по умолчанию 0.1)")
    ap.add_argument("--seed", type=int, default=42, help="Seed для shuffle")
    ap.add_argument("--no-shuffle", action=argparse.BooleanOptionalAction, default=False, help="Отключить перемешивание (делить по порядку)")
    args = ap.parse_args()

    if abs(args.train_ratio + args.dev_ratio + args.test_ratio - 1.0) > 1e-8:
        raise SystemExit("Сумма долей должна быть равна 1.0")

    src_dir = Path(args.src)
    dst_root = Path(args.dst)
    if not src_dir.exists():
        raise SystemExit(f"Источник не найден: {src_dir}")

    present = {name: (src_dir / name).exists() for name in NAME_TO_SUBDIR.keys()}

    handled = set()
    for a_name, b_name in PAIRS:
        a_path = src_dir / a_name
        b_path = src_dir / b_name
        if present.get(a_name) and present.get(b_name):
            process_pair(
                a=a_path, b=b_path, dst_root=dst_root,
                train_r=args.train_ratio, dev_r=args.dev_ratio, test_r=args.test_ratio,
                shuffle=not args.no_shuffle, seed=args.seed
            )
            handled.add(a_name); handled.add(b_name)

    print(f"Done. Datasets root: {dst_root.resolve()}")

if __name__ == "__main__":
    main()
