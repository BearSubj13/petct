#%%
import os
import matplotlib.pyplot as plt

from dataset_with_labels import dcm_to_dict


def load_dcm_series(person_folder):
    dir_list = os.listdir(person_folder)
    if len(dir_list) == 1:
        sub_dir = dir_list[0]
    elif len(dir_list) > 1:
        for sub_dir in dir_list:
            #looks for an non-empty sub_dir
            full_path = os.path.join(person_folder, sub_dir)
            if len(os.listdir(full_path)) > 0:
                break  
    else:
        return None 
    path_dcm = os.path.join(person_folder, sub_dir)
    file_number = 0
    ct_list = []
    
    for dcm_file in os.listdir(path_dcm):
        if dcm_file[:2] == "CT" and dcm_file[-7:] == "dcm.zst":            
            file_path = os.path.join(path_dcm, dcm_file)
            ct_dict = dcm_to_dict(file_path, resolution=64, only_lungs=False)
            if ct_dict is None:
                continue
            else:
                #ct_dict.update({'description':lungs_description})
                ct_list.append(ct_dict)

    return ct_list


if __name__ == "__main__":
  folder = "/ayb/vol3/datasets/pet-ct/DICOM_9801-10000.zip/"
  folder = "/ayb/vol3/datasets/pet-ct/part1/"
  person = os.listdir(folder)[0]
  folder = os.path.join(folder, person)
  ct_list = load_dcm_series(folder)  
  for i, item in enumerate(ct_list):
      if i % 25 == 0:
        image = item["ct"]
        image = image.numpy()
        fig = plt.figure()
        plt.axis('off') 
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()
      #break

# %%
