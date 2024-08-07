import pathlib
import pyconll
import random

def create_train_file(input_path, save_path = 'data.conllu'):
    '''
    Creates a train file from all the files in the input
        file or directory.
    
    Parameters:
    input_path (str): path to the input file or directory
    save_path (str): path to save the train file.
    '''
    save_path = pathlib.Path(save_path)
    save_path.open('w', encoding='utf-8').close()
    
    files = [pathlib.Path(input_path)] if '.conllu' in input_path else pathlib.Path(input_path).rglob("**/*.conllu")
    
    for file in files:
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