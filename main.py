from utils.parser_utils import *
from utils.analyser import Analyser

ud_test_path = "diaparser/ud_test.conllu"
ud_preds_path = "diaparser/ud_preds.conllu"
str_test_path = "diaparser/str_test.conllu"
str_preds_path = "diaparser/str_preds.conllu"


def print_scores(analyzer):
    ucm, lcm, uas, las = analyzer.get_score('ud')
    print(f"UD: UCM: {ucm}%, LCM: {lcm}%, UAS: {uas}%, LAS: {las}%")
    
    ucm, lcm, uas, las = analyzer.get_score('str')
    print(f"STR: UCM: {ucm}%, LCM: {lcm}%, UAS: {uas}%, LAS: {las}%")
    

def main():
    analyzer = Analyser(ud_test_path, ud_preds_path, str_test_path, str_preds_path)
    
    print_scores(analyzer)
    
    analyzer.save_arcs()
    

if __name__ == '__main__':
    main()