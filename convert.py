import pathlib
import pyconll
import xml.etree.ElementTree as ET
from pyconll.unit.token import Token

from utils import get_ud_source, get_str_source
    

def convert_ud_to_ud(read_path, save_path):
    '''
    Convert extended UD fore to simple UD format
    
    Parameters:
    read_path (str): path to the directory with extended UD files.
    save_path (str): path to save the converted corpus.  
    '''
    print("Converting extended UD to base UD...")
    
    # create save directory if non existent
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    
    for file_path in pathlib.Path(read_path).rglob("**/*.conllu"):
        conll = pyconll.load_from_file(file_path)
        
        for sent_id, sent in enumerate(conll, start=1):
            
            ellipsis = 0 
            id_map = {'0': '0'} # handle root dependency
            for new_id, token in enumerate(sent, start=1):
                ellipsis += 1 if token.is_empty_node() else 0
                id_map[token.id] = new_id
            
            if ellipsis:
                for new_id, token in enumerate(sent, start=1):
                    if not token.is_empty_node():
                        token.head = str(id_map[token.head])
                    
                    deps_keys, deps_values = token.deps.keys(), token.deps.values()
                    token.deps = {str(id_map[key]): value for key, value in zip(deps_keys, deps_values)}
                    
                    token.id = str(new_id)
            
            source = get_ud_source(sent)
            sent.set_meta('sent_id', f"{sent_id}_{source}")
        
        # save the converted corpus
        save_file_path = pathlib.Path(save_path) / file_path.name
        with open(save_file_path, 'w', encoding='utf-8') as f:
            conll.write(f)
    
    print("Conversion completed.")

            
            
def convert_str_to_ud(read_path, save_path):
    '''
    Convert all SynTagRus corpus files found in 'read_path'
        directory to Universal Dependencies format.
    
    Parameters:
    read_path (str): path to the directory with SynTagRus files.
    save_path (str): path to save the converted corpus.
    '''
    print("Converting SynTagRus to UD...")
    
    # create save directory if non existent
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # iterate through all .tgt files in the directory 
    for file_path in pathlib.Path(read_path).rglob("**/*.tgt"):
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        source = get_str_source(file_path)
        
        conll_data = []
        for body in root.findall('body'):
            for sent_id, sent in enumerate(body.findall('S'), start=1):
                sentence = []
                link_offset = []
                punct_offset = 0
                
                # punctuation marks before the first word
                for punct in sent.text.strip():
                    if punct.strip():
                        sentence.append(punct.strip())
                        punct_offset += 1
                
                # iterate through all words in the sentence
                for word in sent.findall('W'):
                    sentence.append(word)
                    link_offset.append(punct_offset)
                    
                    # punctuation marks after the word
                    for punct in word.tail.strip():
                        if punct.strip():
                            sentence.append(punct.strip())
                            punct_offset += 1
                
                tokens = []
                punct_offset = 0
                for word_id, word in enumerate(sentence, start=1):
                        
                    if type(word) == ET.Element: # word or ellipsis
                        form = word.text if word.text else '_'
                        upos = (word.get('FEAT')).split()[0]
                        if word.get('DOM') == '_root':
                            head = 0
                            deprel = 'root'
                        else:
                            head = int(word.get('DOM')) + link_offset[int(word.get('DOM')) - 1]
                            deprel = word.get('LINK')
                                    
                    elif type(word) == str: # punctuation mark
                        punct_offset += 1
                        
                        form = word
                        upos = '_'
                        head = '_'
                        deprel = '_'
                    
                    else:
                        raise ValueError('Unknown word type')
                    
                    token_str = f"{word_id}\t{form}\t_\t{upos}\t_\t_\t{head}\t{deprel}\t_\t_\n"
            
                    token = Token(token_str)
                    tokens.append(token)
                
                sentence_str = f"# sent_id = {sent_id}_{source}\n"
                sentence_str += f"# text = {' '.join([token.form for token in tokens if token.form != '_'])}\n"
                for token in tokens:
                    sentence_str += token.conll() + '\n'
            
                conll_data.append(sentence_str)
        
        # save the converted corpus
        file_name = ''.join(file_path.parts[1:])
        file_name = file_name[: file_name.rfind('.tgt')]

        save_file_path = pathlib.Path(save_path) / (file_name+ '.conllu')
        with open(save_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(conll_data))
    
    print("Conversion completed.")