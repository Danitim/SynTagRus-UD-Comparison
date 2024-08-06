import pathlib
import pyconll

from convert.utils import get_source, search_source_name
from convert.utils import match_sentences

def align_package_by_source(package, source_name, str_path, ud_save_path, 
                            str_save_path, unaligned_path, no_source_path,
                            source_aligned, source_unaligned):
    '''
    Align one package of sentences sharing the same source
        name by source name
    
    Parameters:
    package (list): list of UD sentences.
    source_name (str): source name of the package.
    str_path (str): path to the directory with SynTagRus files.
    ud_save_path (Path): path to save the aligned Universal Dependencies corpus.
    str_save_path (Path): path to save the aligned SynTagRus corpus.
    unaligned_path (Path): path to save unaligned sentences.
    no_source_path (Path): path to save sentences with no source found.
    '''
    if search_source_name(source_name, str_path):
        
        # search in the source files
        found_flags = [False for _ in package]
        str_sents = [None for _ in package]
        
        for file_path in pathlib.Path(str_path).rglob("**/*" + source_name + ".conllu"):
            # print(source_name, file_path.name)
            conll = pyconll.load_from_file(file_path)
            
            if len(conll) == len(package):
                for i, (ud_sent, str_sent) in enumerate(zip(package, conll)):
                    match_sent = match_sentences(ud_sent, str_sent)
                    if match_sent:
                        found_flags[i] = True
                        str_sents[i] = match_sent
            else:
                # print("Different sentence count:", len(package), len(conll), source_name)
                idx = [int(sent.meta_value('sent_id').split('_')[0]) - 1 for sent in package]
                for i, sent in enumerate(conll):
                    if i in idx:
                        match_sent = match_sentences(package[idx.index(i)], sent)
                        if match_sent:
                            found_flags[idx.index(i)] = True
                            str_sents[idx.index(i)] = match_sent
                            
        found = sum(found_flags)
        print(f"Found {found} out of {len(package)} sentences in {source_name}.")
                            
                
        if not (file_path in pathlib.Path(str_path).rglob("**/*" + source_name + ".conllu")):
            print("Not found:", source_name)
        if len([file_path for file_path in pathlib.Path(str_path).rglob("**/*" + source_name + ".conllu")]) > 1:
            print("Multiple files found:", source_name)
        
        # save aligned and unaligned sentences
        for ud_sent, str_sent, found in zip(package, str_sents, found_flags):
            if found: # found in source file
                source_aligned += 1
                
                with open(ud_save_path, 'a', encoding='utf-8') as f:
                    f.write(ud_sent.conll() + '\n\n')
                with open(str_save_path, 'a', encoding='utf-8') as f:
                    f.write(str_sent.conll() + '\n\n')
                                
            else: # not found in source file
                source_unaligned += 1                  
                            
                with open(unaligned_path, 'a', encoding='utf-8') as f:
                    f.write(ud_sent.conll() + '\n\n')
                                
    else: # source file not found                  
        with open(no_source_path, 'a', encoding='utf-8') as f:
            for ud_sent in package:
                f.write(ud_sent.conll() + '\n\n')
    
    return source_aligned, source_unaligned


def align_corpora_by_source(save_path, str_path, ud_path,
                ud_save_path="ud_aligned.conllu", str_save_path="str_aligned.conllu",
                unaligned_path="source_unaligned.conllu", no_source_path="source_not_found.conllu"):
    '''
    Align the Universal Dependencies and SynTagRus corpora by the source name.
    
    Parameters:
    save_path (str): path to directory to save aligned corpora files.
    str_path (str): path to the SynTagRus corpus.
    ud_path (str): path to the Universal Dependencies corpus.
    ud_save_path (str): path to save the aligned Universal Dependencies corpus.
    str_save_path (str): path to save the aligned SynTagRus corpus.
    unaligned_path (str): path to save unaligned sentences.
    no_source_path (str): path to save sentences with no source found.
    '''
    print("Starting to align by source name...")
    
    ud_count = 0
    source_aligned = 0
    source_unaligned = 0
    
    # create Paths for save files
    ud_save_path = pathlib.Path(save_path) / ud_save_path
    str_save_path = pathlib.Path(save_path) / str_save_path
    unaligned_path = pathlib.Path(save_path) / unaligned_path
    no_source_path = pathlib.Path(save_path) / no_source_path
    
    # create save files
    ud_save_path.open('w', encoding='utf-8').close()
    str_save_path.open('w', encoding='utf-8').close()
    unaligned_path.open('w', encoding='utf-8').close()
    no_source_path.open('w', encoding='utf-8').close()
    
    for file_path in pathlib.Path(ud_path).rglob("**/*.conllu"):
        conll = pyconll.load_from_file(file_path)
        
        package_source_name = get_source(conll[0])
        package = []
        for ud_sent in conll:
            ud_count += 1
            source_name = get_source(ud_sent)
            
            if source_name != package_source_name:
                source_aligned, source_unaligned = align_package_by_source(package, package_source_name, str_path,
                        ud_save_path, str_save_path, unaligned_path, no_source_path, source_aligned, source_unaligned)
                
                package_source_name = source_name
                package = []
                package.append(ud_sent)
                                
            else:
                package.append(ud_sent)
                
        # align the last package
        source_aligned, source_unaligned = align_package_by_source(package, package_source_name, str_path,
                ud_save_path, str_save_path, unaligned_path, no_source_path, source_aligned, source_unaligned)
            
    source_not_found = ud_count - source_aligned - source_unaligned
                
    # print statistics
    print("Finished aligning by source name.")
    print(f"UD sentences: {ud_count}")
    print("Source aligned:", source_aligned, ", ", round(source_aligned / ud_count * 100, 2), "%")
    print("Source unaligned:", source_unaligned, ", ", round(source_unaligned / ud_count * 100, 2), "%")
    print("Source not found:", source_not_found, ", ", round(source_not_found / ud_count * 100, 2), "%")