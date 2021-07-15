import os
import sys
import pytesseract
import pdf2image
from .striprtf import rtf_to_text


#dst_dir = "texts"
#src_dir = "/ayb/vol3/datasets/pet-ct/part0/990004795/70003837/"

def extract_lungs_description(src_dir):
    is_pdf_avaliable = any([".pdf" in x for x in os.listdir(src_dir)])
    rtf_file = [x for x in os.listdir(src_dir) if ".rtf" in x]

    if is_pdf_avaliable:
        text = ''
        pdf_files = os.popen('find %s -name "*.pdf"' % src_dir)
        for i, line in enumerate(os.popen('find %s -name "*.pdf"' % src_dir)):
            fpath = line.strip()
            image = pdf2image.convert_from_path(fpath)
            for img in pdf2image.convert_from_path(fpath):
                    text += pytesseract.image_to_string(img, lang="eng+rus")
            break
    elif rtf_file:
        rtf_file = [x for x in os.listdir(src_dir) if ".rtf" in x]
        rtf_file = os.path.join(src_dir, rtf_file[0])
        with open(rtf_file, 'r') as file:
            text = file.read()
        text = rtf_to_text(text)
    else:
        return None

    pos = text.find("Органы грудной клетки:")
    if pos == -1:
        return None
    pos2 = text.find('Органы брюшной полости:', pos)
    if pos2 == -1 or pos2 < pos:
        return None
    return text[pos:pos2] 


def extract_lungs_from_txt(text_file_path):
    with open(text_file_path, 'r', encoding='windows-1251') as f:
        text = f.read()
    
    pos = text.find("Органы грудной клетки:")
    if pos == -1:
        return None
    pos2 = text.find('Органы брюшной полости:', pos)
    if pos2 == -1 or pos2 < pos:
        return None
    return text[pos:pos2] 


if __name__ == "__main__":
    # dcm_path = "/ayb/vol3/datasets/pet-ct/part01/"
    # for person in os.listdir(dcm_path):
    #     person_folder = os.path.join(dcm_path, person)
    #     dir_list = os.listdir(person_folder)
    #     if len(dir_list) == 1:
    #         sub_dir = dir_list[0]
    #     elif len(dir_list) > 1:
    #         for sub_dir in dir_list:
    #             #looks for an non-empty sub_dir
    #             full_path = os.path.join(person_folder, sub_dir)
    #             if len(os.listdir(full_path)) > 0:
    #                 break  

    #     path_dcm = os.path.join(person_folder, sub_dir)
    #     lungs_description = extract_lungs_description(path_dcm)
    #     print(lungs_description )

    text_file_path = "/ayb/vol4/sftp/user23/upload/Ростов/AA00277002/заключение.txt"
    extract_lungs_from_txt(text_file_path)