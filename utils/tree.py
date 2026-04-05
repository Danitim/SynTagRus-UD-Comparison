#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import pyconll
import tqdm

def is_basic_token(tok) -> bool:
    tid = getattr(tok, 'id', None)
    return isinstance(tid, str) and tid.isdigit()

def check_sentence(sent):
    errs = []
    nodes = [tok for tok in sent if is_basic_token(tok)]
    ids = {tok.id for tok in nodes}

    if not ids:
        return errs

    invalid_heads = []
    for tok in nodes:
        h = tok.head
        if h in (None, ''):
            invalid_heads.append((tok.id, h))
        elif h != '0' and h not in ids:
            invalid_heads.append((tok.id, h))
    if invalid_heads:
        errs.append(f"Invalid heads: {invalid_heads}")

    roots = [tok.id for tok in nodes if tok.head == '0']
    if len(roots) != 1:
        errs.append(f"Root count != 1 (roots={roots})")

    visited_global = set()
    for tok in nodes:
        if tok.id in visited_global:
            continue
        seen = set()
        cur = tok
        path_ids = []
        ok_path = False
        while True:
            if cur.id in seen:
                errs.append(f"Cycle detected at id={cur.id} (path: {'->'.join(path_ids+[cur.id])})")
                break
            seen.add(cur.id)
            path_ids.append(cur.id)
            h = cur.head
            if h == '0':
                ok_path = True
                break
            if h not in ids:
                errs.append(f"Broken parent chain from id={tok.id}: head {h} not in ids")
                break
            if h == cur.id:
                errs.append(f"Self-loop at id={cur.id}")
                break
            parent = next(n for n in nodes if n.id == h)
            cur = parent
        if ok_path:
            visited_global.update(seen)

    if len(visited_global) != len(ids):
        missing = sorted(ids - visited_global, key=int)
        errs.append(f"Unreachable from root (by following heads to 0): {missing}")

    return errs

def main():
    ap = argparse.ArgumentParser(description="Check that each sentence in a CoNLL-U file is a tree.")
    ap.add_argument("conllu_path", help="Path to .conllu file")
    args = ap.parse_args()

    conll = pyconll.load_from_file(args.conllu_path)

    total = 0
    bad = 0
    for sent in tqdm.tqdm(conll, desc="Checking sentences", unit="sent"):
        total += 1
        sid = sent.meta_value('sent_id') or f"(idx:{total})"
        errs = check_sentence(sent)
        if errs:
            bad += 1
            print(f"[FAIL] sent_id = {sid}")
            for e in errs:
                print("  -", e)

    print("\nSummary:")
    print(f"  sentences checked: {total}")
    print(f"  ok:   {total - bad}")
    print(f"  fail: {bad}")

    sys.exit(1 if bad > 0 else 0)

if __name__ == "__main__":
    main()
