import pathlib
import pyconll
import tqdm

from convert.utils import get_source, search_source_name
from convert.utils import match_sentences
from convert.normalize import normalize_str_sentence_to_ud
from utils.tree import check_sentence

def _infer_split_name_from_filename(filename: str) -> str:
    name = filename.lower()
    if "train" in name:
        return "train"
    elif "dev" in name:
        return "dev"
    elif "test" in name:
        return "test"
    else:
        return "unknown"

def align_package_by_source(package, source_name, str_path, ud_save_path,
                            str_save_path, unaligned_path, no_source_path,
                            source_aligned, source_unaligned, retry,
                            not_tree_path, split_name):
    files = search_source_name(source_name, str_path)
    
    if files:

        found_flags = [False for _ in package]
        str_sents = [None for _ in package]

        for file_path in files:
            conll = pyconll.load_from_file(file_path)

            if len(conll) == len(package):
                for i, (ud_sent, str_sent) in enumerate(zip(package, conll)):
                    match_sent = normalize_str_sentence_to_ud(ud_sent, str_sent)
                    if match_sent:
                        found_flags[i] = True
                        str_sents[i] = match_sent
            else:
                ids = [sent.meta_value('sent_id').split('_')[0] for sent in package]

                for sent in conll:
                    sid = sent.meta_value('sent_id').split('_')[0]
                    if sid in ids:
                        match_sent = normalize_str_sentence_to_ud(package[ids.index(sid)], sent)
                        if match_sent:
                            found_flags[ids.index(sid)] = True
                            str_sents[ids.index(sid)] = match_sent

        found = sum(found_flags)
        print(f"Found {found} out of {len(package)} sentences in {source_name}.")

        for ud_sent, str_sent, found in zip(package, str_sents, found_flags):
            if found:
                source_aligned += 1
                
                if split_name:
                    ud_sent.set_meta("split", split_name)
                    str_sent.set_meta("split", split_name)
                
                try:
                    errs = check_sentence(str_sent)
                except Exception as e:
                    errs = [f"tree_check_exception: {e!r}"]
                if errs:
                    try:
                        with open(not_tree_path, 'a', encoding='utf-8') as nf:
                            sid = ud_sent.meta_value('sent_id') or ''
                            nf.write(f"sent_id = {sid}\n")
                            for e in errs:
                                nf.write(f"- {e}\n")
                            nf.write("\nUD:\n")
                            nf.write(ud_sent.conll() + "\n")
                            nf.write("\nSTR:\n")
                            nf.write(str_sent.conll() + "\n")
                            nf.write("-" * 60 + "\n")
                    except Exception:
                        pass
                
                with open(ud_save_path, 'a', encoding='utf-8') as f:
                    f.write(ud_sent.conll() + '\n\n')
                with open(str_save_path, 'a', encoding='utf-8') as f:
                    f.write(str_sent.conll() + '\n\n')
            else:
                source_unaligned += 1
                with open(unaligned_path, 'a', encoding='utf-8') as f:
                    f.write(ud_sent.conll() + '\n\n')

    else:
        print("Source file not found for:", source_name)
        with open(no_source_path, 'a', encoding='utf-8') as f:
            for ud_sent in package:
                f.write(ud_sent.conll() + '\n\n')

    return source_aligned, source_unaligned


def align_corpora_by_source(save_path, str_path, ud_path,
    ud_save_path="ud_aligned.conllu", str_save_path="str_aligned.conllu",
    unaligned_path="source_unaligned.conllu", no_source_path="source_not_found.conllu",
    retry: bool = False,
    retry_input: str = "source_unaligned.conllu",
    retry_output: str = "source_unaligned_remaining.conllu",
    not_tree_path: str = "not_tree.log"):
    print("Starting to align by source name..." + (" (RETRY MODE)" if retry else ""))


    save_path = pathlib.Path(save_path)
    ud_save_path = save_path / ud_save_path
    str_save_path = save_path / str_save_path
    unaligned_path = save_path / unaligned_path
    no_source_path = save_path / no_source_path
    retry_input_path = save_path / retry_input
    retry_output_path = save_path / retry_output

    ud_count = 0
    source_aligned = 0
    source_unaligned = 0

    if retry:
        retry_output_path.open('w', encoding='utf-8').close()
        if not retry_input_path.exists():
            print(f"[retry] Input file not found: {retry_input_path}")
            return
        ud_conll = pyconll.load_from_file(retry_input_path)
        ud_count = len(ud_conll)

        buckets = {}
        for ud_sent in ud_conll:
            sname = get_source(ud_sent)
            buckets.setdefault(sname, []).append(ud_sent)

        for source_name, package in buckets.items():
            source_aligned, source_unaligned = align_package_by_source(
                package,
                source_name,
                str_path,
                ud_save_path,
                str_save_path,
                retry_output_path,
                no_source_path,
                source_aligned,
                source_unaligned,
                retry,
                not_tree_path
            )

        source_not_found = ud_count - source_aligned - source_unaligned

    else:
        ud_save_path.open('w', encoding='utf-8').close()
        str_save_path.open('w', encoding='utf-8').close()
        unaligned_path.open('w', encoding='utf-8').close()
        no_source_path.open('w', encoding='utf-8').close()

        for file_path in pathlib.Path(ud_path).rglob("**/*.conllu"):
            conll = pyconll.load_from_file(file_path)
            
            split_name = _infer_split_name_from_filename(file_path.name)
            print(f"Processing {file_path.name} (split={split_name})")

            package_source_name = get_source(conll[0])
            package = []
            for ud_sent in conll:
                ud_count += 1
                source_name = get_source(ud_sent)

                if source_name != package_source_name:
                    source_aligned, source_unaligned = align_package_by_source(
                        package,
                        package_source_name,
                        str_path,
                        ud_save_path,
                        str_save_path,
                        unaligned_path,
                        no_source_path,
                        source_aligned,
                        source_unaligned,
                        retry,
                        not_tree_path,
                        split_name
                    )
                    package_source_name = source_name
                    package = [ud_sent]
                else:
                    package.append(ud_sent)

            source_aligned, source_unaligned = align_package_by_source(
                package,
                package_source_name,
                str_path,
                ud_save_path,
                str_save_path,
                unaligned_path,
                no_source_path,
                source_aligned,
                source_unaligned,
                retry,
                not_tree_path,
                split_name
            )

        source_not_found = ud_count - source_aligned - source_unaligned

    print("Finished aligning by source name." + (" (RETRY MODE)" if retry else ""))
    print(f"UD sentences processed: {ud_count}")
    print("Source aligned:", source_aligned, ", ", round((source_aligned / ud_count * 100) if ud_count else 0.0, 2), "%")
    print("Source unaligned:", source_unaligned, ", ", round((source_unaligned / ud_count * 100) if ud_count else 0.0, 2), "%")
    print("Source not found:", source_not_found, ", ", round((source_not_found / ud_count * 100) if ud_count else 0.0, 2), "%")
    if retry:
        print(f"Remaining unaligned after retry -> {retry_output_path.name}")