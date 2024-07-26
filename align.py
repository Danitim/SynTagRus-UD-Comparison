import pathlib
import pyconll
from tqdm import tqdm

ud_aligned_name = "ud_aligned.conllu"
str_aligned_name = "str_aligned.conllu"
source_unaligned_name = "source_unaligned.conllu"
tmp_unaligned_name = "tmp_unaligned.conllu"
globally_unaligned_name = "globally_unaligned.conllu"

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

def align_package(package, source_name, str_path, ud_save_path, str_save_path, 
                  source_unaligned_path, tmp_unaligned_path,
                  source_aligned=0, source_unaligned=0):
    '''
    Align one package of sentences sharing the same source
        name by source name
    
    Parameters:
    package (list): list of UD sentences.
    source_name (str): source name of the package.
    str_path (str): path to the directory with SynTagRus files.
    ud_save_path (Path): path to save the aligned UD sentences.
    str_save_path (Path): path to save the aligned SynTagRus sentences.
    source_unaligned_path (Path): path to save the unaligned UD sentences.
    tmp_unaligned_path (Path): path to save the temporary unaligned UD sentences.
    '''
    if search_source_name(source_name, str_path):
        found_flags, str_sents = align_by_source_name(package, source_name, str_path)
                    
        for ud_sent, str_sent, found in zip(package, str_sents, found_flags):
            if found: # found in source file
                source_aligned += 1
                            
                with open(ud_save_path, 'a', encoding='utf-8') as f:
                    f.write(ud_sent.conll() + '\n\n')
                with open(str_save_path, 'a', encoding='utf-8') as f:
                    f.write(str_sent.conll() + '\n\n')
                                
            else: # not found in source file
                source_unaligned += 1                  
                            
                with open(source_unaligned_path, 'a', encoding='utf-8') as f:
                    f.write(ud_sent.conll() + '\n\n')
                                
    else: # source file not found                  
        with open(tmp_unaligned_path, 'a', encoding='utf-8') as f:
            for ud_sent in package:
                f.write(ud_sent.conll() + '\n\n')
    
    return source_aligned, source_unaligned


def align_by_source_name(package, source_name, str_path):
    '''
    Search for the corresponding sentences in SynTagRus corpus
    
    Parameters:
    package (list): list of UD sentences.
    source_name (str): source name of the package.
    str_path (str): path to the directory with SynTagRus files.
    
    Returns:
    found_flags (list): list of boolean values indicating whether
        the corresponding sentence was found.
    str_sents (list): list of corresponding SynTagRus sentences.
    '''
    found_flags = [False for _ in package]
    str_sents = [None for _ in package]
    
    for file_path in pathlib.Path(str_path).rglob("**/*.conllu"):
        if source_name in file_path.name:
            conll = pyconll.load_from_file(file_path)
            
            for str_sent in conll:
                for i, ud_sent in enumerate(package):
                    if compare_sentences(ud_sent, str_sent):
                        found_flags[i] = True
                        str_sents[i] = str_sent
                        break
    
    return found_flags, str_sents

def align_globally(ud_sent, str_path):
    '''
    Align UD sentence globally with SynTagRus sentences
    
    Parameters:
    ud_sent (pyconll.unit.sentence.Sentence): UD sentence.
    str_path (str): path to the directory with SynTagRus files.
    
    Returns:
    found (bool): whether the sentence was aligned.
    str_sent (pyconll.unit.sentence.Sentence): corresponding SynTagRus sentence.
    '''
    found = False
    str_sent = None
    
    for file_path in pathlib.Path(str_path).rglob("**/*.conllu"):
        conll = pyconll.load_from_file(file_path)
            
        for sent in conll:
            if compare_sentences(ud_sent, sent):
                found = True
                str_sent = sent
                break
        
        if found:
            break
    
    return found, str_sent
                    

def align_corpora(ud_path, str_path, save_path):
    '''
    Align two corpora by iterating through all SynTagRus UD sentences
        and searching the corresponding sentence in SynTagRus corpus.
        All aligned sentences are saved in the corresponding files.
        
    Parameters:
    ud_path (str): path to the directory with UD SynTagRus files.
    str_path (str): path to the directory with SynTagRus files.
    save_path (str): path to save the aligned corpora.
    '''
    ud_count = 0
    source_aligned = 0
    source_unaligned = 0
    global_aligned = 0
    globally_unaligned= 0
    
    # create save directory if non existent
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # create aligned corpora files
    ud_save_path = pathlib.Path(save_path) / ud_aligned_name
    open(ud_save_path, 'w', encoding='utf-8').close()
    
    str_save_path = pathlib.Path(save_path) / str_aligned_name
    open(str_save_path, 'w', encoding='utf-8').close()
    
    # create file for source unaligned sentences
    source_unaligned_path = pathlib.Path(save_path) / source_unaligned_name
    open(source_unaligned_path, 'w', encoding='utf-8').close()
    
    # create temporary file for unaligned sentences
    tmp_unaligned_path = pathlib.Path(save_path) / tmp_unaligned_name
    open(tmp_unaligned_path, 'w', encoding='utf-8').close()
    
    
    # align by source name
    print("Starting to align by source name...")
    for file_path in pathlib.Path(ud_path).rglob("**/*.conllu"):
        conll = pyconll.load_from_file(file_path)
        
        package_source_name = get_source(conll[0])
        package = []
        for ud_sent in conll:
            ud_count += 1
            source_name = get_source(ud_sent)
            
            if source_name != package_source_name:
                source_aligned, source_unaligned = align_package(package,
                        package_source_name, str_path, ud_save_path, str_save_path,
                        source_unaligned_path, tmp_unaligned_path, source_aligned, source_unaligned)
                
                package_source_name = source_name
                package = []
                package.append(ud_sent)
                                
            else:
                package.append(ud_sent)
                
        # align the last package
        source_aligned, source_unaligned = align_package(package, package_source_name,
                str_path, ud_save_path, str_save_path, source_unaligned_path, tmp_unaligned_path,
                source_aligned, source_unaligned)
            
    source_not_found = ud_count - source_aligned - source_unaligned
                
    # print statistics
    print("Finished aligning by source name.")
    print(f"UD sentences: {ud_count}")
    print("Source aligned:", source_aligned, ", ", round(source_aligned / ud_count * 100, 2), "%")
    print("Source unaligned:", source_unaligned, ", ", round(source_unaligned / ud_count * 100, 2), "%")
    print("Source not found:", ud_count - source_aligned - source_unaligned, ", ", 
          round(source_not_found / ud_count * 100, 2), "%")
    print('-' * 50)
    
    
    # create file for unaligned sentences
    globally_unaligned_path = pathlib.Path(save_path) / globally_unaligned_name
    open(globally_unaligned_path, 'w', encoding='utf-8').close()
            
    # global alignment
    print("Starting to align globally...")
    tmp_unaligned_conll = pyconll.load_from_file(tmp_unaligned_path)
    for ud_sent, i in zip(tmp_unaligned_conll, tqdm(range(source_not_found), desc="Global alignment")):
        found, str_sent = align_globally(ud_sent, str_path)
        
        if found:
            global_aligned += 1
            
            with open(ud_save_path, 'a', encoding='utf-8') as f:
                f.write(ud_sent.conll() + '\n\n')
            with open(str_save_path, 'a', encoding='utf-8') as f:
                f.write(str_sent.conll() + '\n\n')
        else:
            globally_unaligned += 1
            
            with open(globally_unaligned_path, 'a', encoding='utf-8') as f:
                f.write(ud_sent.conll() + '\n\n')
    
    
    # print statistics
    print("Finished aligning globally.")
    print(f"UD sentences: {source_not_found}")
    print("Globally aligned:", global_aligned, ", ", round(global_aligned / ud_count * 100, 2), "%")
    print("Globally unaligned:", globally_unaligned, ", ", round((ud_count - global_aligned) / ud_count * 100, 2), "%")
    print('-' * 50)
                
                
    # print statistics
    print("Finished aligning corpora.")
    print(f"UD sentences: {ud_count}")
    print("Aligned:", source_aligned + global_aligned, ", ", 
          round((source_aligned + global_aligned) / ud_count * 100, 2), "%")
    print("Unaligned:", source_unaligned + globally_unaligned, ", ",
            round((source_unaligned + globally_unaligned) / ud_count * 100, 2), "%")
    
    # delete temporary file
    tmp_unaligned_path.unlink()