import pyconll
import pandas as pd

from utils.score_utils import ucm_score, lcm_score, uas_score, las_score

class Analyser:
    def __init__(self,
                ud_test_path="diaparser/ud_test.conllu",
                ud_preds_path="diaparser/ud_preds.conllu",
                str_test_path="diaparser/str_test.conllu",
                str_preds_path="diaparser/str_preds.conllu"):
        
        self.ud_test_path = ud_test_path
        self.ud_preds_path = ud_preds_path
        self.str_test_path = str_test_path
        self.str_preds_path = str_preds_path

        self.ud_test = pyconll.load_from_file(self.ud_test_path)
        self.ud_preds = pyconll.load_from_file(self.ud_preds_path)
        self.str_test = pyconll.load_from_file(self.str_test_path)
        self.str_preds = pyconll.load_from_file(self.str_preds_path)
        
    def get_score(self, mode='ud', punct=False):
        '''
        Returns UCM, LCM, UAS, LAS scores
            
        Parameters:
            mode (str): 'ud' or 'str'
            punct (bool: False): whether to include punctuation in the score
            
        Returns:
            tuple: (UCM, LCM, UAS, LAS), percentage
        '''
        match mode:
            case 'ud':
                test = self.ud_test
                preds = self.ud_preds
            case 'str':
                test = self.str_test
                preds = self.str_preds
            case _:
                raise ValueError("Mode should be 'ud' or 'str'")
                
        sent_total = len(test)
        word_total = 0
        ucm, lcm, uas, las = 0, 0, 0, 0
            
        for gold, pred in zip(test, preds):
            word_total += sum([1 for word in gold if punct or word.upos != 'PUNCT'])
            
            ucm += ucm_score(gold, pred)
            lcm += lcm_score(gold, pred)
            uas += uas_score(gold, pred)
            las += las_score(gold, pred)
            
        ucm = round(ucm / sent_total * 100, 2)
        lcm = round(lcm / sent_total * 100, 2)
        uas = round(uas / word_total * 100, 2)
        las = round(las / word_total * 100, 2)
            
        return (ucm, lcm, uas, las)
    
    
    def save_arcs(self, save_path="results.csv"):
        '''
        Saves the arcs to a csv file
        '''
        data = {
            'ud_test_head': [word.head for sent in self.ud_test for word in sent],
            'ud_test_deprel': [word.deprel for sent in self.ud_test for word in sent],
            'ud_preds_head': [word.head for sent in self.ud_preds for word in sent],
            'ud_preds_deprel': [word.deprel for sent in self.ud_preds for word in sent],
            'str_test_head': [word.head for sent in self.str_test for word in sent],
            'str_test_deprel': [word.deprel for sent in self.str_test for word in sent],
            'str_preds_head': [word.head for sent in self.str_preds for word in sent],
            'str_preds_deprel': [word.deprel for sent in self.str_preds for word in sent],
        }
        
        df = pd.DataFrame(data)
        
        df.to_csv(save_path, index=False)
            