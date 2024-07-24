import os
import re
import string

from pyconll.unit.token import Token

import xml.etree.ElementTree as ET

source_pattern = r"([a-zA-Zа-яА-Я]+(?:_[a-zA-Zа-яА-Я]+)*)"

def read_str_file(root, name):
    '''
    Read a single SynTagRus file.
    
    Parameters:
    root (str): path to the directory with the file.
    name (str): name of the file.
    
    Returns:
    list: list of dictionaries with the following keys:
        'sentence' (list): list of words in the sentence.
            Each word is either a string (punctuation mark)
            or an ElementTree.Element (word or ellipsis);
        'source_name' (str): name of the source file.
    '''
    file_path = os.path.join(root, name)
    sentences = []
    
    tree = ET.parse(file_path)
    root = tree.getroot()

    for body in root.findall('body'):
        for sent in body.findall('S'):
            
            # collect all text from the sentence (including punctuation marks)
            text = []
            for elem in sent.itertext():
                for part in elem.split():
                    if re.match(r'\w+', part):
                        text.append(part)
                    else:
                        for punct in part:
                            text.append(punct)
            text = [elem for elem in text if elem]
                
            
            # collect all words from the sentence and insert punctuation marks inbetween
            sentence = []
            text_iter = 0
            link_offset = 0
            for word in sent.findall('W'):
                # punctuation marks before the word
                while (text_iter < len(text)) and (text[text_iter] in string.punctuation):
                    sentence.append(text[text_iter])
                    text_iter += 1
                    link_offset += 1
                
                # adjust the indices of the words after the inserted punctuation marks
                if not word.get('DOM') == '_root':
                    word.set('DOM', str(int(word.get('DOM')) + link_offset))
                    
                sentence.append(word)
                
                text_iter += 1 if word.text else 0 # in case of an ellipsis
                
            # punctuation marks after the last word
            while text_iter < len(text):
                sentence.append(text[text_iter])
                text_iter += 1
                
            # print(text)
            # print([word.text if type(word) == ET.Element else word for word in sentence])
                        
            source_name = re.search(source_pattern, name).group(1)
                
            sentences.append({'sentence': sentence, 'source_name': source_name})
    
    return sentences

def read_str_corpus(path, file_extension=".tgt"):
    '''
    Read SynTagRus corpus from folder.
    
    Parameters:
    path (str): path to the directory with files.
    file_extension (str): extension of the files to read.
    
    Returns:
    list: list of dictionaries with the following keys:
        'sentence' (list): list of words in the sentence.
            Each word is either a string (punctuation mark)
            or an ElementTree.Element (word or ellipsis);
        'source_name' (str): name of the source file.
    '''
    sentences = []

    for root, _, files in os.walk(path):
        for name in files:
            if name.endswith(file_extension):
                file_sentences = read_str_file(root, name)
                sentences.extend(file_sentences)

    return sentences

def convert_str_to_ud(read_path, save_path):
    '''
    Convert all SynTagRus corpus files found in 'read_path'
        directory to Univerasl Dependencies format.
    
    Parameters:
    read_path (str): path to the directory with SynTagRus files.
    save_path (str): path to save the converted corpus.
    '''
    conll_data = []
    
    sentences = read_str_corpus(read_path)
    
    for sent_id, sent in enumerate(sentences, start=1):
        tokens = []

        for word_id, word in enumerate(sent['sentence'], start=1):
            if type(word) == ET.Element: # word or ellipsis
                upos = (word.get('FEAT')).split()[0]    
                form = word.text if word.text else '_'
                    
                if word.get('DOM') == '_root': # root dependency
                    head = 0
                    deprel = 'root'
                else: # other dependencies
                    head = int(word.get('DOM'))
                    deprel = word.get('LINK')
                    
            elif type(word) == str: # punctuation mark
                form = word
                upos = 'PUNCT'
                head = '_' # actual head will be determined later after corpus comparison
                deprel = 'punct'
            else:
                raise ValueError(f"Unexpected type: {type(word)}")
            
            token_str = f"{word_id}\t{form}\t_\t{upos}\t_\t_\t{head}\t{deprel}\t_\t_\n"
            
            token = Token(token_str)
            tokens.append(token)
            
        sentence_str = f"# sent_id = {sent_id}_{sent['source_name']}\n"
        sentence_str += f"# text = {' '.join([token.form for token in tokens if token.form != '_'])}\n"
        for token in tokens:
            sentence_str += token.conll() + '\n'
        
        conll_data.append(sentence_str) 
    
    with open(save_path, 'w', encoding='utf-8') as f:
        for sent in conll_data:
            f.write(sent + '\n')