import argparse
import pathlib

from convert.convert import convert_str_to_ud, convert_ud_to_ud
from convert.source_align import align_corpora_by_source
from convert.global_align import align_corpora_globally
from convert.punctuation import align_punctuation

str_path = "Corpora/SynTagRus"
str_converted_path = "Corpora/STR_converted"
ud_path = "Corpora/UD_SynTagRus"
ud_converted_path = "Corpora/UD_converted"
aligned_path = "Aligned"

def main():
    parser = argparse.ArgumentParser(description='Convert SynTagRus to UD and align the corpora.')
   
    parser.add_argument('mode', type=str, help='Mode of the script: convert, align or align SynTagRus punctuation (punct).')
    parser.add_argument('--corpus', type=str, help='Corpus to convert: STR or UD.', default="STR")
    parser.add_argument('--align_mode', type=str, help='Mode of alignment: source or global.', default="source")
   
    parser.add_argument('--str_path', type=str, help='Path to SynTagRus corpus.', default=str_path)
    parser.add_argument('--str_converted_path', type=str, help='Path to converted SynTagRus corpus.', default=str_converted_path)
    parser.add_argument('--ud_path', type=str, help='Path to UD SynTagRus corpus.', default=ud_path)
    parser.add_argument('--ud_converted_path', type=str, help='Path to converted UD SynTagRus corpus.', default=ud_converted_path)
   
    parser.add_argument('--aligned_path', type=str, help='Path to save aligned corpora files.', default=aligned_path)

   
    args = parser.parse_args()
   
    if (args.mode == "convert"):
        if (args.corpus == "STR"):
            convert_str_to_ud(args.str_path, args.str_converted_path)
        elif (args.corpus == "UD"):
            convert_ud_to_ud(args.ud_path, args.ud_converted_path)
        else:
            print("Invalid corpus.")
            
    elif (args.mode == "align"):
        # create save directory if non existent
        pathlib.Path(args.aligned_path).mkdir(parents=True, exist_ok=True)
      
        if (args.align_mode == "source"):
            align_corpora_by_source(args.aligned_path, args.str_converted_path, args.ud_converted_path)
        elif (args.align_mode == "global"):
            align_corpora_globally(args.aligned_path, args.str_converted_path)
        else:
            print("Invalid alignment mode.")

    elif (args.mode == "punct"):
        align_punctuation(args.aligned_path)
                    
            
if __name__ == '__main__':
    main()