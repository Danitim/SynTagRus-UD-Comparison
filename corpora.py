import argparse
import pathlib

from convert.convert import convert_str_to_ud, convert_ud_to_ud
from convert.source_align import align_corpora_by_source
from convert.global_align import align_corpora_globally

str_path = "Corpora/SynTagRus"
str_converted_path = "Corpora/STR_converted"
ud_path = "Corpora/UD_SynTagRus"
ud_converted_path = "Corpora/UD_converted"
aligned_path = "Aligned"

def main():
    parser = argparse.ArgumentParser(description='Convert SynTagRus to UD and align the corpora.')

    parser.add_argument('mode', type=str, help='Mode of the script: convert or align.')
    parser.add_argument('--corpus', type=str, help='Corpus to convert: STR or UD.', default="STR")
    parser.add_argument('--align_mode', type=str, help='Mode of alignment: source or global.', default="source")

    parser.add_argument('--str_path', type=str, default=str_path)
    parser.add_argument('--str_converted_path', type=str, default=str_converted_path)
    parser.add_argument('--ud_path', type=str, default=ud_path)
    parser.add_argument('--ud_converted_path', type=str, default=ud_converted_path)

    parser.add_argument('--aligned_path', type=str, default=aligned_path)

    # новый флаг
    parser.add_argument('--retry', action='store_true',
                        help='Retry only for previously unaligned-by-source sentences.')

    # имена файлов можно переопределять при желании
    parser.add_argument('--unaligned_in', type=str, default="source_unaligned.conllu",
                        help='When --retry, take UD sentences from this file (inside aligned_path).')
    parser.add_argument('--unaligned_out', type=str, default="source_unaligned_remaining.conllu",
                        help='When --retry, write remaining unaligned to this file (inside aligned_path).')

    args = parser.parse_args()

    if args.mode == "convert":
        pathlib.Path(args.aligned_path).mkdir(parents=True, exist_ok=True)
        if args.corpus == "STR":
            convert_str_to_ud(args.str_path, args.str_converted_path)
        elif args.corpus == "UD":
            convert_ud_to_ud(args.ud_path, args.ud_converted_path)
        else:
            print("Invalid corpus.")

    elif args.mode == "align":
        pathlib.Path(args.aligned_path).mkdir(parents=True, exist_ok=True)

        if args.align_mode == "source":
            align_corpora_by_source(
                save_path=args.aligned_path,
                str_path=args.str_converted_path,
                ud_path=args.ud_converted_path,
                retry=args.retry,
                retry_input=args.unaligned_in,
                retry_output=args.unaligned_out,
            )

        elif args.align_mode == "global":
            align_corpora_globally(
                save_path=args.aligned_path,
                str_path=args.str_converted_path,
                ud_path="source_not_found.conllu"
            )
        else:
            print("Invalid alignment mode.")
    else:
        print("Invalid mode.")

if __name__ == '__main__':
    main()