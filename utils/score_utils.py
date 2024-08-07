from pyconll.unit.sentence import Sentence

def ucm_score(gold: Sentence, pred: Sentence, punct=False) -> bool:
    '''
    Returns wheter the UCM score is True or False
    
    Parameters:
        gold (Sentence): gold sentence
        pred (Sentence): predicted sentence
        punct (bool: False): whether to include punctuation in the score
    
    Returns:
        bool: True if UCM is correct, False otherwise
    '''
    return all([gold_word.head == pred_word.head for gold_word, pred_word in zip(gold, pred) if
                punct or gold_word.upos != 'PUNCT'])

def lcm_score(gold: Sentence, pred: Sentence, punct=False) -> bool:
    '''
    Returns wheter the LCM score is True or False
    
    Parameters:
        gold (Sentence): gold sentence
        pred (Sentence): predicted sentence
        punct (bool: False): whether to include punctuation in the score
    
    Returns:
        bool: True if LCM is correct, False otherwise
    '''
    ucm = ucm_score(gold, pred)
    return ucm and all([gold_word.deprel == pred_word.deprel for gold_word, pred_word in zip(gold, pred) if
                        punct or gold_word.upos != 'PUNCT'])

def uas_score(gold: Sentence, pred: Sentence, punct=False) -> int:
    '''
    Returns the number of correct UAS dependencies
    
    Parameters:
        gold (Sentence): gold sentence
        pred (Sentence): predicted sentence
        punct (bool: False): whether to include punctuation in the score
    
    Returns:
        int: number of correct UAS dependencies
    '''
    return sum([1 for gold_word, pred_word in zip(gold, pred) if
                gold_word.head == pred_word.head and (punct or gold_word.upos != 'PUNCT')])

def las_score(gold: Sentence, pred: Sentence, punct=False) -> int:
    '''
    Returns the number of correct LAS dependencies
    
    Parameters:
        gold (Sentence): gold sentence
        pred (Sentence): predicted sentence
        punct (bool: False): whether to include punctuation in the score
    
    Returns:
        int: number of correct LAS dependencies
    '''
    return sum([1 for gold_word, pred_word in zip(gold, pred) if
                gold_word.head == pred_word.head and gold_word.deprel == pred_word.deprel
                and (punct or gold_word.upos != 'PUNCT')])