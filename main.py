from parser_utils import *

test_path = 'diaparser/ud_test.conllu'
preds_path = 'diaparser/ud_preds.conllu'

def main():
    create_train_file('Corpora/STR_converted', 'data.conllu')
    
    train_dev_test_split(0.85, 0.05, -1, 'data.conllu', 'diaparser/', 'str')
    
    # ucm, lcm, uas, las = ellipsis_score(test_path, preds_path)
    # print(f'UCM: {ucm}%  LCM: {lcm}%  UAS: {uas}%  LAS: {las}%')

if __name__ == '__main__':
    main()