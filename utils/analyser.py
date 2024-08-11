import pyconll
import pandas as pd

from utils.score_utils import ucm_score, lcm_score, uas_score, las_score
from utils.parser_utils import ellipsis_flag, get_dependents

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
            'word': [word.form.lower() if word.form else '_' for sent in self.ud_test for word in sent],
            'ud_test_head': [word.head for sent in self.ud_test for word in sent],
            'ud_test_deprel': [word.deprel for sent in self.ud_test for word in sent],
            'ud_preds_head': [word.head for sent in self.ud_preds for word in sent],
            'ud_preds_deprel': [word.deprel for sent in self.ud_preds for word in sent],
            'str_test_head': [word.head for sent in self.str_test for word in sent],
            'str_test_deprel': [word.deprel for sent in self.str_test for word in sent],
            'str_preds_head': [word.head for sent in self.str_preds for word in sent],
            'str_preds_deprel': [word.deprel for sent in self.str_preds for word in sent],
            'ellipsis': [0 if word.upos and word.upos != '_' else 1 for sent in self.ud_test for word in sent],
            'text': [sent.meta_value('text') for sent in self.ud_test for word in sent],
            'sent_id': [sent.meta_value('sent_id') for sent in self.ud_test for word in sent]
        }
        
        df = pd.DataFrame(data)
        
        df.to_csv(save_path, index=False, sep=',')
        
    
    def save_ellipsis(self, save_path="ellipsis.csv"):
        '''
        Saves the ellipsis sentences to a csv file
        '''
        data = {
            'id': [],
            'ud_test_head': [],
            'ud_test_deprel': [],
            'ud_test_deps_id': [],
            'ud_test_deps_deprel': [],
            'ud_pred_head': [],
            'ud_pred_deprel': [],
            'ud_pred_deps_id': [],
            'ud_pred_deps_deprel': [],
            'str_test_head': [],
            'str_test_deprel': [],
            'str_test_deps_id': [],
            'str_test_deps_deprel': [],
            'str_pred_head': [],
            'str_pred_deprel': [],
            'str_pred_deps_id': [],
            'str_pred_deps_deprel': [],
            'text': [],
            'sent_id': []           
        }
        
        for ud_test_sent, ud_preds_sent, str_test_sent, str_preds_sent in zip(self.ud_test, self.ud_preds, self.str_test, self.str_preds):
            if not ellipsis_flag(ud_test_sent):
                continue
            
            ellipsis = {}
            for word in ud_test_sent:
                if not word.upos or word.upos == '_':
                    ellipsis[int(word.id)] = []
            
            for ellipsis_id in list(ellipsis.keys()):
                data['id'].append(ellipsis_id)
                
                data['ud_test_head'].append(ud_test_sent[ellipsis_id-1].head)
                data['ud_test_deprel'].append(ud_test_sent[ellipsis_id-1].deprel)
                data['ud_test_deps_id'].append([id for id in get_dependents(ud_test_sent, ellipsis_id)])
                data['ud_test_deps_deprel'].append([ud_test_sent[id-1].deprel for id in get_dependents(ud_test_sent, ellipsis_id)])
                
                data['ud_pred_head'].append(ud_preds_sent[ellipsis_id-1].head)
                data['ud_pred_deprel'].append(ud_preds_sent[ellipsis_id-1].deprel)
                data['ud_pred_deps_id'].append([id for id in get_dependents(ud_preds_sent, ellipsis_id)])
                data['ud_pred_deps_deprel'].append([ud_preds_sent[id-1].deprel for id in get_dependents(ud_preds_sent, ellipsis_id)])
                
                data['str_test_head'].append(str_test_sent[ellipsis_id-1].head)
                data['str_test_deprel'].append(str_test_sent[ellipsis_id-1].deprel)
                data['str_test_deps_id'].append([id for id in get_dependents(str_test_sent, ellipsis_id)])
                data['str_test_deps_deprel'].append([str_test_sent[id-1].deprel for id in get_dependents(str_test_sent, ellipsis_id)])
                
                data['str_pred_head'].append(str_preds_sent[ellipsis_id-1].head)
                data['str_pred_deprel'].append(str_preds_sent[ellipsis_id-1].deprel)
                data['str_pred_deps_id'].append([id for id in get_dependents(str_preds_sent, ellipsis_id)])
                data['str_pred_deps_deprel'].append([str_preds_sent[id-1].deprel for id in get_dependents(str_preds_sent, ellipsis_id)])
                
                data['text'].append(ud_test_sent.meta_value('text'))
                data['sent_id'].append(ud_test_sent.meta_value('sent_id'))
            
        
        df = pd.DataFrame(data)
        
        df.to_csv(save_path, index=False, sep=',')
            