from pyconll.unit.sentence import Sentence

def ucm_score(gold: Sentence, pred: Sentence, punct=False) -> bool:
    return all([gold_word.head == pred_word.head for gold_word, pred_word in zip(gold, pred) if
                punct or gold_word.upos != 'PUNCT'])

def lcm_score(gold: Sentence, pred: Sentence, punct=False) -> bool:
    ucm = ucm_score(gold, pred)
    return ucm and all([gold_word.deprel == pred_word.deprel for gold_word, pred_word in zip(gold, pred) if
                        punct or gold_word.upos != 'PUNCT'])

def uas_score(gold: Sentence, pred: Sentence, punct=False) -> int:
    return sum([1 for gold_word, pred_word in zip(gold, pred) if
                gold_word.head == pred_word.head and (punct or gold_word.upos != 'PUNCT')])

def las_score(gold: Sentence, pred: Sentence, punct=False) -> int:
    return sum([1 for gold_word, pred_word in zip(gold, pred) if
                gold_word.head == pred_word.head and gold_word.deprel == pred_word.deprel
                and (punct or gold_word.upos != 'PUNCT')])