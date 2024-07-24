from str_to_ud import convert_str_to_ud

ud_path = "UD_SynTagRus"
str_path = "SynTagRus"

def main():
   convert_str_to_ud(str_path, "converted_str.conllu")
                    
            
if __name__ == '__main__':
    main()