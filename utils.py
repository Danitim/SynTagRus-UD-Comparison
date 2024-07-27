import pyconll
import pathlib

def get_ud_source(sent):
    '''
    Extract the source name from the UD sentence.
    
    Parameters:
    sent (pyconll.unit.sentence.Sentence): sentence.
    '''
    source = sent.meta_value('sent_id')
    source = source[: source.rfind('.xml')]
    
    while not source[-1].isalpha():
        source = source[:-1]

    return source


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
    Checks wheter the source name is present in the directory
    
    Parameters:
    source_name (str): source name to search for.
    path (str): path to the directory.
    '''
    for file_path in pathlib.Path(path).rglob("**/*.conllu"):
        if file_path.name.lower().find(source_name.lower()) != -1:
            return True
        
    return False

def compare_sentences(ud_sent, str_sent):
    '''
    Compare two sentences for complete word form match
    
    Parameters:
    ud_sent (pyconll.unit.sentence.Sentence): UD sentence.
    str_sent (pyconll.unit.sentence.Sentence): SynTagRus sentence.
    '''
    if len(ud_sent) != len(str_sent):
        return False
    
    for ud_token, str_token in zip(ud_sent, str_sent):
        if ud_token.form != str_token.form:
            return False
        
    return True