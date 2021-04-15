#%%
import os
from shutil import rmtree
import json
from tqdm import tqdm

import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_file
from pydicom.pixel_data_handlers.util import apply_modality_lut
import zstandard as zstd
from io import BytesIO

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from create_dataset import normalization_window, lung_files
from utils.pdf_extraction import extract_lungs_description


class LungsLabeled(Dataset):
    """
    terminate: int or False - stop add new values to dataset if index > terminate
    """
    def __init__(self, dataset_path, terminate=False, load_memory=False):
        super().__init__()
        self.load_memory = load_memory
        #files indexes for a person in the list - "person id":[list of indexes]
        self.person_label_dict = dict()
        #a dictionary contains all information about a person
        self.person_index_dict = dict()
        self.image_file_list = []
        self.label_list = []

        for person_index, person_id in enumerate(tqdm(os.listdir(dataset_path))):
            person_folder = os.path.join(dataset_path, person_id)
            label_folder = os.path.join(person_folder, 'labeles')
            image_folder = os.path.join(person_folder, 'images')
            assert len(os.listdir(label_folder)) == len(os.listdir(image_folder))

            temp_dict_json = dict()
            for file_json in os.listdir(label_folder):
                key_number = int(file_json[:-5])
                json_path = os.path.join(label_folder, file_json)
                with open(json_path, 'r') as fp:
                    label_dict = json.load(fp)
                    if label_dict["sex"] == "M":
                       label_dict["sex"] = 0
                    else:
                       label_dict["sex"] = 1  

                temp_dict_json.update({key_number:label_dict})

            temp_dict_image = dict() 
            for image_file in os.listdir(image_folder):
                key_number = int(image_file[:-3])
                image_path = os.path.join(image_folder, image_file)
                if load_memory:
                    image_path = torch.load(image_path)
                temp_dict_image.update({key_number:image_path})

            index_list = []
            for i in range(len(os.listdir(label_folder))):
                self.image_file_list.append(temp_dict_image[i])
                self.label_list.append(temp_dict_json[i])
                index_list.append(len(self.image_file_list) - 1)

            self.person_index_dict.update({person_id:index_list})
            person_inform = label_dict
            del person_inform['slice']
            del person_inform['position']
            self.person_label_dict.update({person_id:person_inform})

            if terminate:
                if person_index > terminate:
                    return None
            

        assert len(self.image_file_list) == len(self.label_list)
            
    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, index):
        label = self.label_list[index]
        if self.load_memory:
            image = self.image_file_list[index]
        else:
            image = torch.load(self.image_file_list[index])
        return image, label

    # def get_person_item(person_id, index):
    #     list_of_indexes = self.person_index_dict[person_id]
    #     return self.__getitem__(list_of_indexes[index])


def dcm_to_dict(file_path, resolution=64, verbose=True, only_lungs=True):
    zd = zstd.ZstdDecompressor()
    compressed = open(file_path, "rb").read()
    try:
        data = zd.decompress(compressed)
        ds = pydicom.dcmread(BytesIO(data))
    except:
        if verbose:
            print("error in reading", file_path)
        return None 

    if only_lungs and 'Lung' not in ds.SeriesDescription:
        return None

    arr = ds.get('pixel_array', None)#ds.pixel_array
    if arr is None:
        if verbose:
            print("No image in", file_path)
        return None
    hu =    (arr, ds)
    arr = normalization_window(hu, ds)
    hu = torch.FloatTensor(hu)
    if hu.max() - hu.min() == 0:
        return None
    hu = (hu - hu.min()) / (hu.max() - hu.min())
    
    if resolution != 512:
        image_tensor = hu
        image_tensor = image_tensor.unsqueeze(0)#.unsqueeze(0)
        image_tensor = F.interpolate(image_tensor, size=resolution)
        image_tensor = image_tensor.permute(0, 2, 1)
        image_tensor = F.interpolate(image_tensor, size=resolution) 
        image_tensor = image_tensor.permute(0, 2, 1)
        image_tensor = image_tensor.squeeze()

    slice_loc = ds.get('SliceLocation', None)
    sex = ds.get('PatientSex', None)
    weight = ds.get('PatientWeight', None)
    weight = float(weight)
    length = ds.get('PatientSize', None)
    length = float(length)
    patient_id = ds.get('PatientID', None)
    age = ds.get('PatientAge', None)
    age = int(age[:-1])
    if slice_loc is None:
        print("No slice in", file_path)
        return None
    else:
        slice_loc = float(slice_loc)

    return_dict = {"slice":slice_loc, "image":image_tensor, "sex":sex, "weight":weight, \
                    "length":length, "id":patient_id, "age":age, "position":None, "file":file_path}
    return return_dict


def save_dcm_series(person_folder, dataset_path, type_of_image="CT"):
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

    lungs_description = extract_lungs_description(path_dcm)
    if lungs_description is None:
        return None
    
    for dcm_file in os.listdir(path_dcm):
        if dcm_file[:2] == type_of_image and dcm_file[-7:] == "dcm.zst":
            file_path = os.path.join(path_dcm, dcm_file)
            ct_dict = dcm_to_dict(file_path, resolution=64)
            if ct_dict is None:
                continue
            else:
                ct_dict.update({'description':lungs_description})
                ct_list.append(ct_dict)

    #normalize slice from 0 to 1
    if len(ct_list) > 0:
        min_slice = min(ct_list, key=lambda x: x["slice"] )["slice"]
        max_slice = max(ct_list, key=lambda x: x["slice"] )["slice"]
        d = max_slice - min_slice
        for i, ct_dict in enumerate(ct_list):
            position = (ct_dict["slice"] - min_slice) / d
            ct_list[i].update({"position":position}) 
        ct_list = sorted(ct_list, key=lambda x: x["position"])
    else:
        print("empty list:", path_dcm)
        return None

    person_folder = os.path.join(dataset_path, ct_list[0]["id"])
    if os.path.exists(person_folder):
        rmtree(person_folder)
    os.mkdir(person_folder)
    image_folder = os.path.join(person_folder, "images")
    label_folder = os.path.join(person_folder, "labeles")
    os.mkdir(image_folder)
    os.mkdir(label_folder)

    for i, ct_dict in enumerate(ct_list):
        image_path = os.path.join(image_folder, str(i) + ".pt")
        torch.save(ct_dict["image"], image_path)
        del ct_dict["image"]
        label_path = os.path.join(label_folder, str(i) + ".json")
        with open(label_path, 'w') as fp:
            json.dump(ct_dict, fp)
     
 
if __name__ == "__main__":
    dcm_path = "/ayb/vol3/datasets/pet-ct/part01/"
    dataset_path = "/ayb/vol1/kruzhilov/datasets/labeled_lungs_description/val"
    #for person in os.listdir(dcm_path):
        #old_dir = os.listdir("/ayb/vol1/kruzhilov/datasets/labeled_lungs/train")
        #if person not in old_dir:
        #person_folder = os.path.join(dcm_path, person)
        #save_dcm_series(person_folder, dataset_path, type_of_image="PI")

    lung_dataset = LungsLabeled(dataset_path, terminate=5, load_memory=False)
    #image, label = lung_dataset.__getitem__(3300)
    key = list(lung_dataset.person_index_dict.keys())[0]
    print(lung_dataset.person_label_dict[key])
    # plt.imshow(image, cmap=plt.cm.gray)
    # plt.show()

#error in reading /ayb/vol3/datasets/pet-ct/part0/990004795/70003837/CT.1.2.840.113619.2.290.3.279707939.332.1521085394.707.149.dcm.zst
#error in reading /ayb/vol3/datasets/pet-ct/part0/990004453/70003535/CT.1.2.840.113619.2.290.3.279707939.240.1517454262.499.101.dcm.zst
#error in reading /ayb/vol3/datasets/pet-ct/part0/990004454/70003559/CT.1.2.840.113619.2.290.3.279707939.108.1517886235.971.27.dcm.zst
#error in reading /ayb/vol3/datasets/pet-ct/part0/70000605/70001651/CT.1.2.840.113619.2.290.3.279707939.455.1490932492.808.154.dcm.zst
#error in reading /ayb/vol3/datasets/pet-ct/part0/60001077/70001770/CT.1.2.840.113619.2.290.3.279707939.211.1492660019.821.496.dcm.zst
#error in reading /ayb/vol3/datasets/pet-ct/part0/990003070/70002296/CT.1.2.840.113619.2.290.3.279707939.191.1502247828.441.123.dcm.zst
#error in reading /ayb/vol3/datasets/pet-ct/part0/990002678/70002032/CT.1.2.840.113619.2.290.3.279707939.115.1498100816.569.627.dcm.zst
#error in reading /ayb/vol3/datasets/pet-ct/part0/70000933/70003739/CT.1.2.840.113619.2.290.3.279707939.212.1519789302.352.281.dcm.zst
#error in reading /ayb/vol3/datasets/pet-ct/part0/990004185/70003302/CT.1.2.840.113619.2.290.3.279707939.128.1514259302.703.430.dcm.zst
#error in reading /ayb/vol3/datasets/pet-ct/part0/990003728/70003066/CT.1.2.840.113619.2.290.3.279707939.195.1512012422.930.5.dcm.zst
#error in reading /ayb/vol3/datasets/pet-ct/part0/990002194/70001685/CT.1.2.840.113619.2.290.3.279707939.785.1491537445.8.39.dcm.zst
#error in reading /ayb/vol3/datasets/pet-ct/part0/990001714/70001285/CT.1.2.840.113619.2.290.3.279707939.629.1484886025.832.42.dcm.zst
#error in reading /ayb/vol3/datasets/pet-ct/part0/990004559/70003629/CT.1.2.840.113619.2.25.4.83425147.1519020509.932.dcm.zst
#error in reading /ayb/vol3/datasets/pet-ct/part0/990003898/70003046/CT.1.2.840.113619.2.290.3.279707939.285.1511926941.527.421.dcm.zst
#error in reading /ayb/vol3/datasets/pet-ct/part0/990004892/70003965/CT.1.2.840.113619.2.290.3.279707939.130.1522725787.881.246.dcm.zst
#error in reading /ayb/vol3/datasets/pet-ct/part0/990002275/70001745/CT.1.2.840.113619.2.290.3.279707939.948.1492488545.585.397.dcm.zst
#error in reading /ayb/vol3/datasets/pet-ct/part0/990001848/70001374/CT.1.2.840.113619.2.290.3.279707939.145.1486177721.911.754.dcm.zst
#error in reading /ayb/vol3/datasets/pet-ct/part0/990002436/70001879/CT.1.2.840.113619.2.290.3.279707939.347.1495166516.684.207.dcm.zstempty list: /ayb/vol3/datasets/pet-ct/part0/70000551/70001396

#empty list: /ayb/vol3/datasets/pet-ct/part0/990001963/70001489
#empty list: /ayb/vol3/datasets/pet-ct/part0/70000551/70001396

# %%
