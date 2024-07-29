import pathlib
import pyconll
from tqdm import tqdm

def align_punctuation(aligned_path, str_aligned_path="str_aligned.conllu", ud_aligned_path="ud_aligned.conllu"):
    '''
    Since the punctuation in SynTagRus format is not tokenized,
        we need to align the punctuation in the UD format using
        the punctuation from the UD corpus for each aligned
        sentence.
        
    Parameters:
        aligned_path (str): Path to the aligned corpora.
    '''
    print("Aligning the punctuation dependencies between corpora...")
    
    str_path = pathlib.Path(aligned_path) / str_aligned_path
    ud_path = pathlib.Path(aligned_path) / ud_aligned_path
    
    str = pyconll.load_from_file(str_path)
    ud = pyconll.load_from_file(ud_path)
    
    assert len(str) == len(ud), "The corpora must have the same number of sentences."
    for i, str_sent, ud_sent in zip(tqdm(range(len(str))), str, ud):
        
        assert len(str_sent) == len(ud_sent), "The sentences must have the same number of tokens."
        for str_token, ud_token in zip(str_sent, ud_sent):
            if ud_token.upos == 'PUNCT':
                str_token.head = ud_token.head
                str_token.deprel = ud_token.deprel
                
    with open(str_path, 'w', encoding='utf-8') as f:
        str.write(f)
                
    print("Punctuation alignment completed.")