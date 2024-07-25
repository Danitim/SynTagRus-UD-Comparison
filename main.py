from convert import *

ud_path = "UD_SynTagRus"
str_path = "SynTagRus"

def main():
   convert_str_to_ud(str_path, "STR_converted")
   convert_ud_to_ud(ud_path, "UD_converted")
                    
            
if __name__ == '__main__':
    main()