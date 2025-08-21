import pyconll
import pathlib

from pyconll.unit.sentence import Sentence
from pyconll.unit.token import Token
from string import punctuation

def get_ud_source(sent):
    '''
    Extract the source name and id from the UD sentence.
    
    Parameters:
    sent (pyconll.unit.sentence.Sentence): sentence.
    
    Returns:
    source (str): source name.
    id (str): sentence id.
    '''
    source = sent.meta_value('sent_id')
    id = source[source.rfind('.xml')+5:]
    source = source[: source.rfind('.xml')]
    
    # while not source[-1].isalpha():
    #     source = source[:-1]

    return source, id


def get_str_source(file_path):
    '''
    Extract the source name from the SynTagRus file path.
    
    Parameters:
    file_path (str): path to the SynTagRus file.
    
    Returns:
    str: source name.
    '''
    source = ''.join(file_path.parts[1:])
    source = source[: source.rfind('.tgt')]
    
    while not source[-1].isalpha():
        source = source[:-1]
        
    return source


def get_source(sent):
    '''
    Extract source name from the sentence
    
    Parameters:
    sent (pyconll.unit.sentence.Sentence): sentence.
    
    Returns:
    source_name (str): source name of the sentence.
    '''
    first_underscore = sent.meta_value('sent_id').find('_')
    return sent.meta_value('sent_id')[first_underscore + 1:]


def search_source_name(source_name, path):
    '''
    Search for files with the given source name in the specified path.
    
    Parameters:
    source_name (str): source name to search for.
    path (str): path to the directory.
    
    Returns:
    list: list of file paths containing the source name.
    '''
    files = []
    for file_path in pathlib.Path(path).rglob("**/*.conllu"):
        if file_path.name.lower().find(source_name.lower()) != -1:
            files.append(file_path)
    return files

def match_sentences(ud_sent, str_sent):
    '''
    Compare two sentences for complete word form match without
        counting punctuation.
    
    Parameters:
    ud_sent (pyconll.unit.sentence.Sentence): UD sentence.
    str_sent (pyconll.unit.sentence.Sentence): SynTagRus sentence.
    
    Returns:
    str_sent (pyconll.unit.sentence.Sentence): SynTagRus sentence with
        fixed punctuation, if sentences match, None otherwise.
    '''
    def strip_punct(token):
        return token.form.strip(punctuation + '…').lower() if (token.form and token.form != '_') else '_'
    
    def set_form(token, form) -> Token:
        token_id = token.id
        upos = token.upos
        head, deprel= token.head, token.deprel
        
        return Token(f"{token_id}\t{form}\t_\t{upos}\t_\t_\t{head}\t{deprel}\t_\t_")
    
    
    ud_words = [token for token in ud_sent if token.upos != 'PUNCT']
    str_words = [token for token in str_sent if token.upos != 'PUNCT']
    
    if len(ud_words) != len(str_words):
        if len([word for word in ud_words if strip_punct(word) != '_']) != len([word for word in str_words if strip_punct(word) != '_']):
            with open("length_unmatched.txt", 'a', encoding='utf-8') as f:
                f.write("sent_id = " + str_sent.meta_value('sent_id') + '\n')
                f.write("UD: " + ' '.join([strip_punct(token) for token in ud_sent]) + '\n')
                f.write("STR: " + ' '.join([strip_punct(token) for token in str_sent]) + '\n')
                f.write("-" * 100 + '\n')
        return None
    
    unmatched = []
    for idx, (ud_word, str_word) in enumerate(zip(ud_words, str_words)):
        if strip_punct(ud_word) != strip_punct(str_word):
            str_words[idx] = set_form(str_word, ud_word.form)
            unmatched.append(strip_punct(ud_word) + '!=' + strip_punct(str_word))
            
    if len(unmatched) > 0:
        ud_word_forms = [strip_punct(token) for token in ud_sent]
        str_word_forms = [strip_punct(token) for token in str_sent]
        with open("unmatched.txt", 'a', encoding='utf-8') as f:
            f.write("sent_id = " + str_sent.meta_value('sent_id') + '\n')
            for unmatched_pair in unmatched:
                f.write(unmatched_pair + '\n')
            f.write("UD: " + ' '.join(ud_word_forms) + '\n')
            f.write("STR: " + ' '.join(str_word_forms) + '\n')
            f.write("-" * 100 + '\n')
        
    
    new_sent = Sentence(ud_sent.conll())
    str_index = 0
    link_map = {'0': '0'}
    for token in new_sent:
        if token.upos != 'PUNCT':
            str_token = str_words[str_index]
            str_index += 1
            
            link_map[str_token.id] = token.id
            
    str_index = 0
    for token in new_sent:
        if token.upos != 'PUNCT':
            str_token = str_words[str_index]
            str_index += 1
            
            token.upos = str_token.upos
            token.xpos = None
            token.head = link_map[str_token.head]
            token.deprel = str_token.deprel
            
        token.lemma = None
        token.feats = {}
        token.misc = {}
            
    return new_sent
    

def restore_ellipsis(sent, dependents):
    '''
    Restore the ellipsis nodes in the sentence
    
    Parameters:
    sent (pyconll.unit.sentence.Sentence): sentence.
    dependents (dict): dictionary of dependents of each ellipsis node.
        Each element is a tuple of (token, deprel).
    '''
    count = len(dependents)
    
    while count > 0:
        for head, deps in dependents.items():
            ellipsis = sent[head]
            
            ids = [dep[0].id for dep in deps]
            heads = [dep[0].head for dep in deps]
            
            # continue if at least one head in deps[2] is not restored yet
            if any(dep[0].head is None for dep in deps) or ellipsis.head:
                continue
            
            # attach empty elipsis to its head from deps
            if not deps:
                ellipsis.head = list(ellipsis.deps.keys())[0]
                ellipsis.deprel = list(ellipsis.deps.values())[0][0]
                
                if len(ellipsis.deps) > 1:
                    for h, rel in ellipsis.deps.items():
                        if '.' not in h:
                            ellipsis.head = h
                            ellipsis.deprel = rel[0]
                            break
                
                count -= 1
                continue
            
            # find the promoted node
            ids = [dep[0].id for dep in deps]
            heads = [dep[0].head for dep in deps]
            for h in heads:
                if h not in ids:
                    promoted = sent[ids[heads.index(h)]]
                    break
                
            # attach ellipsis to the real head
            ellipsis.head = promoted.head
            ellipsis.deprel = promoted.deprel
            
            # attach ellipsis dependents to the ellipsis
            for dep in deps:
                dep[0].head = ellipsis.id
                dep[0].deprel = dep[1]
                
            count -= 1
            