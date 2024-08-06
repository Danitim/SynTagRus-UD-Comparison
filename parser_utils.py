import pathlib
import pyconll
import random

def create_train_file(input_path, save_path = 'data.conllu'):
    '''
    Creates a train file from all the files in the input directory.
    
    Parameters:
    input_path (str): path to the input directory.
    save_path (str): path to save the train file.
    '''
    save_path = pathlib.Path(save_path)
    save_path.open('w', encoding='utf-8').close()
    
    for file in pathlib.Path(input_path).rglob("**/*.conllu"):
        conll = pyconll.load_from_file(file)
        
        print(f"Writing {file}")
        with open(save_path, 'a', encoding='utf-8') as f:
            conll.write(f)
    
    print("Saved")
    

def train_dev_test_split(train_size=0.85, dev_size=0.05, data_size=-1,
                         input_path = "Aligned/ud_aligned.conllu",
                         output_path = "diaparser/",
                         corpus_tag = "ud",
                         random_seed = 42):
    '''
    Splits the data into train, dev, and test sets.
    
    Parameters:
    train_size (float): proportion of the data to be used for training.
    dev_size (float): proportion of the data to be used for development.
    data_size (int): number of sentences to be used for training, development,
    and testing; if -1, all data is used.
    input_path (str): path to the input file.
    corpus_tag (str): tag to be added to the train, dev, and test files.
    output_path (str): path to save the train, dev, and test files.    
    '''
    random.seed(random_seed)
    
    path = pathlib.Path(input_path)
    conll = pyconll.load_from_file(path)
    print("Opened")
    
    random.shuffle(conll)
    if (data_size == -1):
        data_size = len(conll)
    
    slice_1 = int(train_size * data_size)
    slice_2 = int((train_size + dev_size) * data_size)
    
    train = conll[:slice_1]
    dev = conll[slice_1:slice_2]
    test = conll[slice_2:data_size]
    print("Split")
    
    train_path = pathlib.Path(output_path) / (corpus_tag + "_train.conllu")
    with open(train_path, "w", encoding='utf-8') as f:
        train.write(f)
    
    dev_path = pathlib.Path(output_path) / (corpus_tag + "_dev.conllu")
    with open(dev_path, "w", encoding='utf-8') as f:
        dev.write(f)
        
    test_path = pathlib.Path(output_path) / (corpus_tag + "_test.conllu")
    with open(test_path, "w", encoding='utf-8') as f:
        test.write(f)
    print("Saved")
    
    
def ellipsis_score(test_path, preds_path, save=True,
                   save_test='diaparser/ud_ell_test.conllu',
                   save_preds='diaparser/ud_ell_preds.conllu'):
    '''
    Calculates the UCM, LCM, UAS, and LAS scores for ellipsis, if save
        flag is set, saves the sentences with ellipsis in test and preds.
    
    Parameters:
    test_path (str): path to the test file.
    preds_path (str): path to the predicted file.
    save (bool): flag to save the sentences with ellipsis in test and preds.
    save_test (str): path to save the test file with ellipsis.
    save_preds (str): path to save the predicted file with ellipsis.
    
    Returns:
    ucm (float): unlabeled complete match score.
    lcm (float): labeled complete match score.
    uas (float): unlabeled attachment score.
    las (float): labeled attachment score.
    '''
    ucm, lcm, uas, las = 0, 0, 0, 0
    sent_total, word_total = 0, 0
    
    test = pyconll.load_from_file(test_path)
    preds = pyconll.load_from_file(preds_path)
    
    test_ellipsis = []
    preds_ellipsis = []
    
    assert len(test) == len(preds), 'Different number of sentences in test and preds'
    for test_sent, pred_sent in zip(test, preds):
        ellipsis = False
        
        for token in test_sent:
            if not token.upos:
                ellipsis = True
                break
            
        if ellipsis:
            test_ellipsis.append(test_sent.conll())
            preds_ellipsis.append(pred_sent.conll())
            
            sent_total += 1
            
            ucm_flag, lcm_flag = True, True
            for test_token, pred_token in zip(test_sent, pred_sent):                
                if not test_token.upos:
                    word_total += 1
                    
                    if test_token.head == pred_token.head:
                        uas += 1
                        if test_token.deprel == pred_token.deprel:
                            las += 1
                
                if test_token.head != pred_token.head:
                    ucm_flag = False
                    lcm_flag = False
                elif test_token.deprel != pred_token.deprel:
                    lcm_flag = False
                    
            if ucm_flag:
                ucm += 1
                if lcm_flag:
                    lcm += 1
            
    
    if save:
        save_test_path = pathlib.Path(save_test)
        with open(save_test_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_ellipsis))
            
        save_preds_path = pathlib.Path(save_preds)
        with open(save_preds_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(preds_ellipsis))
            
    ucm = round(ucm / sent_total * 100, 2)
    lcm = round(lcm / sent_total * 100, 2)
    uas = round(uas / word_total * 100, 2)
    las = round(las / word_total * 100, 2)
    
    return ucm, lcm, uas, las