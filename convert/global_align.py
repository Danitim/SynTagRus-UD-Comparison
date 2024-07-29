import pathlib
import pyconll
from tqdm import tqdm

from convert.utils import compare_sentences

def align_corpora_globally(save_path, str_path, ud_path="source_not_found.conllu",
        ud_save_path="ud_aligned.conllu", str_save_path="str_aligned.conllu",
        unaligned_path="globally_unaligned.conllu"):
    '''
    Align the Universal Dependencies and SynTagRus corpora globally,
        by comparing all sentences in the corpora.
        
    Parameters:
    save_path (str): path to directory to save aligned corpora files.
    str_path (str): path to the SynTagRus corpus.
    ud_path (str): path to the Universal Dependencies file within
        the aligned directory.
    ud_save_path (str): path to save the aligned Universal Dependencies corpus.
    str_save_path (str): path to save the aligned SynTagRus corpus.
    unaligned_path (str): path to save unaligned sentences.
    '''
    print("Starting to align globally...")
    
    globally_aligned = 0
    globally_unaligned = 0
    
    # create Paths for save files
    ud_path = pathlib.Path(save_path) / ud_path
    ud_save_path = pathlib.Path(save_path) / ud_save_path
    str_save_path = pathlib.Path(save_path) / str_save_path
    unaligned_path = pathlib.Path(save_path) / unaligned_path
    
    # create save files
    unaligned_path.open('w', encoding='utf-8').close()
    
    ud_conll = pyconll.load_from_file(ud_path)
    ud_count = len(ud_conll)
    
    for i, ud_sent in zip(tqdm(range(ud_count), desc="Global alignment"), ud_conll):
        
        # search for the corresponding sentence in SynTagRus
        found = False
        str_sent = None
        for file_path in pathlib.Path(str_path).rglob("**/*.conllu"):
            conll = pyconll.load_from_file(file_path)
            
            for sent in conll:
                if compare_sentences(ud_sent, sent):
                    found = True
                    str_sent = sent
                    break
        
            if found:
                break
        
        
        # save the aligned (or unaligned) sentences
        if found:
            globally_aligned += 1
            
            with open(ud_save_path, 'a', encoding='utf-8') as f:
                f.write(ud_sent.conll() + '\n\n')
            with open(str_save_path, 'a', encoding='utf-8') as f:
                f.write(str_sent.conll() + '\n\n')
        else:
            globally_unaligned += 1
            
            with open(unaligned_path, 'a', encoding='utf-8') as f:
                f.write(ud_sent.conll() + '\n\n')
    
    
    # print statistics
    print("Finished aligning globally.")
    print(f"UD sentences: {ud_count}")
    print("Globally aligned:", globally_aligned, ", ", round(globally_aligned / ud_count * 100, 2), "%")
    print("Globally unaligned:", globally_unaligned, ", ", round(globally_unaligned / ud_count * 100, 2), "%")