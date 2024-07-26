from convert import *
from align import align_corpora

str_path = "SynTagRus"
str_converted_path = "STR_converted"
ud_path = "UD_SynTagRus"
ud_converted_path = "UD_converted"

def main():
   # convert_str_to_ud(str_path, str_converted_path)
   # convert_ud_to_ud(ud_path, ud_converted_path)
   
   align_corpora(ud_converted_path, str_converted_path, "aligned")
                    
            
if __name__ == '__main__':
    main()