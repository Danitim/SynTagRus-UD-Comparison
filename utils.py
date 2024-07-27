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
            