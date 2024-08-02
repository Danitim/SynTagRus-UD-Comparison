from parser_utils import *

test_path = 'diaparser/ud_test.conllu'
preds_path = 'diaparser/ud_preds.conllu'

def main():
    # train_dev_test_split(0.85, 0.05, -1, 'Aligned/ud_aligned.conllu', 'diaparser/')
    
    ucm, lcm, uas, las = ellipsis_score(test_path, preds_path)
    print(f'UCM: {ucm}%  LCM: {lcm}%  UAS: {uas}%  LAS: {las}%')

if __name__ == '__main__':
    main()