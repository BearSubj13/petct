#%%
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from dataset_with_labels import dcm_to_dict, LungsLabeled, crop_resize_image


def load_dcm_series(person_folder, series_type="LUNG"):
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
    pi_list = []
    slice_list = []
    
    for dcm_file in os.listdir(path_dcm):
        if dcm_file[-7:] == "dcm.zst":            
            file_path = os.path.join(path_dcm, dcm_file)
            ct_dict = dcm_to_dict(file_path, resolution=256, only_lungs=False)
            if ct_dict is None:
                continue
            elif ct_dict["modality"] == "PT":
                #ct_dict.update({'description':lungs_description})
                pi_list.append(ct_dict)
            elif ct_dict["modality"] == "CT" and series_type in ct_dict["series"].upper():
                slice_list.append(ct_dict["slice"])
                ct_list.append(ct_dict)

    min_lung = min(slice_list)
    max_lung = max(slice_list)
    return ct_list, pi_list, min_lung, max_lung


def load_dcm_series_audit(person_folder, series_type="LUNG"):
    sub_dir = "DICOM"
    path_dcm = os.path.join(person_folder, sub_dir)
    if not os.path.isdir(path_dcm):
        warnings.warn("the folder does not exist: "+path_dcm)
        return None
    ct_list = []
    pi_list = []
    slice_list = []
     
    for dcm_file in os.listdir(path_dcm):
        if dcm_file[-3:] == "dcm":   
            file_path = os.path.join(path_dcm, dcm_file)
            pict_dict = dcm_to_dict(file_path, resolution=256, only_lungs=False)
            if pict_dict is None:
                continue
            else:
                if pict_dict["modality"] == "CT":
                    #print(pict_dict["series"])
                    if series_type in pict_dict["series"].upper():
                        ct_list.append(pict_dict)
                        slice_list.append(pict_dict["slice"])
                elif pict_dict["modality"] == "PT":
                    pict_dict = dcm_to_dict(file_path, resolution=256, only_lungs=False, normalize=False)
                    pi_list.append(pict_dict)

    min_lung = min(slice_list)
    max_lung = max(slice_list)
    return ct_list, pi_list, min_lung, max_lung


def draw_file():
    file_path = "/ayb/vol3/datasets/pet-ct/audit/Астрахань/OA00309652/DICOM/1.2.840.113619.2.354.3.2181701635.617.1610275008.899.143.dcm"
    dcm_dict = dcm_to_dict(file_path, resolution=256)
    print(dcm_dict["series"])
    image = dcm_dict["ct"]
    plt.axis("off")
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()


#18.30 taxi
if __name__ == "__main__":
    dataset_path = "/ayb/vol1/kruzhilov/datasets/labeled_lungs_description/labeled_lungs_description_256/deleteme"
    dataset_ctpi = LungsLabeled(dataset_path, resolution=256, terminate=6)
    #OA00309652  RECON 3: LUNG 5 MM
    #/ayb/vol3/datasets/pet-ct/audit/Астрахань/OA00309652/DICOM/1.2.840.113619.2.354.3.2181701635.617.1610275008.899.143.dcm
    i = 150
    print(i)
    ct, pi = dataset_ctpi.__getitem__(i)
    ct2, pi = dataset_ctpi.__getitem__(i + 45)
    image = ct.numpy()
    image2 = (pi/pi.max()).numpy()
    image = np.concatenate([image2, image, np.zeros_like(image2)], axis=0)
    image = np.moveaxis(image, source=[0,1,2], destination=[2,0,1])
    plt.axis('off') 
    plt.imshow(image)
    fig = plt.figure()
    plt.axis('off') 
    plt.imshow(image2.squeeze(0), cmap=plt.cm.gray)

    #draw_file()

  #folder = "/ayb/vol3/datasets/pet-ct/audit/Астрахань/"
#   folder = "/ayb/vol3/datasets/pet-ct/part01/"
#   person = os.listdir(folder)[0]
#   #person = "OA00309652"
#   folder = os.path.join(folder, person)
#   ct_list, pi_list, min_lung, max_lung = load_dcm_series(folder, series_type="CTAC")  
#   ct_list = sorted(ct_list, key=lambda x: x["slice"])
#   for i, item in enumerate(ct_list):
#       if i % 50 == 0:
#         if min_lung < item["slice"] and max_lung > item["slice"]:
#             image = item["ct"]
#             image = image.numpy()
#             #fig = plt.figure()
#             #plt.axis('off') 
#             #plt.imshow(image, cmap=plt.cm.gray)
#             #plt.show()
#             image2 = pi_list[i]["ct"]
#             image2 = image2.numpy()/image2.max()
#             image = np.stack([image2, image, np.zeros_like(image2)], axis=0)
#             image = np.moveaxis(image, source=[0,1,2], destination=[2,0,1])
#             plt.axis('off') 
#             plt.imshow(image, cmap=plt.cm.gray)
#             break
        



# %%
