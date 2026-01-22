#!/usr/bin/env python3
import argparse
import os
from typing import Dict, Iterable, List, Optional, Tuple, Union

CORPORA = ["str", "str-new", "str-old", "ud", "ud-new", "ud-old"]
SPLITS = ["train", "dev", "test"]


def _is_ellipsis(form: str) -> bool:
    return (form is None) or (form == "") or (form == "_")


def _sent_id_from_comment(line: str) -> Optional[str]:
    if not line.startswith("#"):
        return None
    line_lower = line.lower()
    if not line_lower.startswith("# sent_id"):
        return None
    parts = line.split("=", 1)
    if len(parts) != 2:
        return None
    return parts[1].strip()


TokenLine = List[str]
SentenceLine = Union[str, TokenLine]


class Sentence:
    def __init__(self, sent_id: Optional[str], lines: List[SentenceLine], index: int):
        self.sent_id = sent_id
        self.lines = lines
        self.index = index

    def token_map(self) -> Dict[str, TokenLine]:
        tokens: Dict[str, TokenLine] = {}
        for item in self.lines:
            if isinstance(item, list):
                tok_id = item[0]
                if tok_id.isdigit():
                    tokens[tok_id] = item
        return tokens


def _iter_sentences(path: str) -> Iterable[Sentence]:
    sentences: List[Sentence] = []
    current: List[SentenceLine] = []
    sent_id: Optional[str] = None
    index = 0

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line.strip() == "":
                if current:
                    sentences.append(Sentence(sent_id, current, index))
                    index += 1
                current = []
                sent_id = None
                continue

            found_id = _sent_id_from_comment(line)
            if found_id is not None:
                sent_id = found_id
                current.append(line)
                continue

            cols = line.split("\t")
            if len(cols) == 10:
                current.append(cols)
            else:
                current.append(line)

    if current:
        sentences.append(Sentence(sent_id, current, index))
    return sentences


def _build_reference_by_id(sentences: Iterable[Sentence]) -> Tuple[Dict[str, Sentence], int]:
    by_id: Dict[str, Sentence] = {}
    missing_id = 0
    for sent in sentences:
        if not sent.sent_id:
            missing_id += 1
            continue
        if sent.sent_id not in by_id:
            by_id[sent.sent_id] = sent
    return by_id, missing_id


def _apply_ellipsis(reference: Sentence, target: Sentence) -> int:
    ref_tokens = reference.token_map()
    replaced = 0
    for i, item in enumerate(target.lines):
        if not isinstance(item, list):
            continue
        tok_id = item[0]
        if not tok_id.isdigit():
            continue
        ref = ref_tokens.get(tok_id)
        if ref is None:
            continue
        ref_form = ref[1]
        if _is_ellipsis(item[1]):
            if item[1] != ref_form:
                item[1] = ref_form
                _set_ellipsis_flag(item)
                replaced += 1
    return replaced


def _set_ellipsis_flag(token: TokenLine) -> None:
    misc_index = 9
    marker = "Ellipsis=Yes"
    misc = token[misc_index] if len(token) > misc_index else "_"
    if misc in ("", "_"):
        token[misc_index] = marker
        return
    if marker in misc.split("|"):
        return
    token[misc_index] = "{}|{}".format(misc, marker)


def _write_sentences(path: str, sentences: Iterable[Sentence]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for item in sent.lines:
                if isinstance(item, list):
                    f.write("\t".join(item) + "\n")
                else:
                    f.write(item + "\n")
            f.write("\n")


def restore_file(ref_by_id: Dict[str, Sentence], target_path: str, output_path: str) -> None:
    tgt_sentences = list(_iter_sentences(target_path))

    replaced_total = 0
    matched_id = 0
    missing = 0

    for sent in tgt_sentences:
        if sent.sent_id and sent.sent_id in ref_by_id:
            matched_id += 1
            replaced_total += _apply_ellipsis(ref_by_id[sent.sent_id], sent)
        else:
            missing += 1

    _write_sentences(output_path, tgt_sentences)
    print(
        "[OK] {} -> {} (sent_id matched: {}, missing: {}, ellipsis restored: {})".format(
            target_path, output_path, matched_id, missing, replaced_total
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Restore ellipsis tokens in transformed corpora using sent_id alignment."
    )
    ap.add_argument(
        "--reference-root",
        required=True,
        help="Root with reference train/dev/test (ellipsis present, not split by corpus).",
    )
    ap.add_argument("--input-root", required=True, help="Root with transformed corpora (ellipsis filled).")
    ap.add_argument("--output-root", default=None, help="Output root (default: <input-root>_ellipsis).")
    ap.add_argument("--in-place", action="store_true", help="Overwrite files under input-root.")
    args = ap.parse_args()

    input_root = args.input_root
    if args.in_place:
        output_root = input_root
    else:
        output_root = args.output_root or (input_root.rstrip("/\\") + "_ellipsis")

    reference_by_id: Dict[str, Sentence] = {}
    missing_ref_ids = 0
    for split in SPLITS:
        ref_path = os.path.join(args.reference_root, f"{split}.conllu")
        if not os.path.isfile(ref_path):
            print(f"[SKIP] Missing reference: {ref_path}")
            continue
        ref_sentences = list(_iter_sentences(ref_path))
        by_id, missing = _build_reference_by_id(ref_sentences)
        missing_ref_ids += missing
        for sent_id, sent in by_id.items():
            if sent_id not in reference_by_id:
                reference_by_id[sent_id] = sent

    if missing_ref_ids:
        print(f"[WARN] Reference sentences without sent_id: {missing_ref_ids}")

    for corpus in CORPORA:
        for split in SPLITS:
            in_path = os.path.join(input_root, corpus, f"{split}.conllu")
            out_path = os.path.join(output_root, corpus, f"{split}.conllu")

            if not os.path.isfile(in_path):
                print(f"[SKIP] Missing input: {in_path}")
                continue

            restore_file(reference_by_id, in_path, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
