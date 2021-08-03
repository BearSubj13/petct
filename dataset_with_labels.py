#%%
# constants for the Hubert normalization https://radiopaedia.org/articles/windowing-ct
#"CTAC 3.75 Thick" и "WB 3D MAC" coinsedes, 'CTAC 3.75 mm' - Балашиха
import os
from shutil import rmtree
import json
import warnings
from tqdm import tqdm
import copy
#from shlex import quote

import matplotlib.pyplot as plt
import pydicom
#from pydicom.data import get_testdata_file
from pydicom.pixel_data_handlers.util import apply_modality_lut
import zstandard as zstd
from io import BytesIO

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from create_dataset import normalization_window #, lung_files
from utils.pdf_extraction import extract_lungs_description, extract_lungs_from_txt
from utils.pi_files import rescale_image, read_pi_files, interpolate_image


class LungsLabeled(Dataset):
    """
    dataset_path: string - path to the dataset
    terminate: int or False - stop add new values to dataset if index > terminate
    load_memory: boolean - if True loads images to memory
    """
    def __init__(self, dataset_path, resolution=64, terminate=False, load_memory=False, load_labels=False):
        super().__init__()
        self.load_memory = load_memory
        self.resolution = resolution
        #files indexes for a person in the list - "person id":[list of indexes]
        self.person_label_dict = dict()
        #a dictionary contains all information about a person e.g. weight, length, sex, ct description
        self.person_index_dict = dict()
        self.ct_file_list = []
        self.pi_file_list = []
        self.label_list = []
        self.load_labels = load_labels

        for person_index, person_id in enumerate(tqdm(os.listdir(dataset_path))):
            person_folder = os.path.join(dataset_path, person_id)
            label_folder = os.path.join(person_folder, 'labeles')
            ct_folder = os.path.join(person_folder, 'ct')
            pi_folder = os.path.join(person_folder, 'pi')
            assert len(os.listdir(label_folder)) == len(os.listdir(ct_folder))
            assert len(os.listdir(label_folder)) == len(os.listdir(pi_folder))

            #a label for a person with description
            label_path = os.path.join(person_folder, "labeles.json")
            with open(label_path, 'r') as fp:
                person_label = json.load(fp)
            self.person_label_dict.update({person_id:person_label})

            #add labels
            temp_dict_json = dict()
            if load_labels:
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

            #add ct images
            temp_dict_ct = dict() 
            for image_file in os.listdir(ct_folder):
                key_number = int(image_file[:-3])
                image_path = os.path.join(ct_folder, image_file)
                if load_memory:
                    image_path = torch.load(image_path)
                    image_path = rescale_image(image_path, output_size=resolution)
                temp_dict_ct.update({key_number:image_path})

            #add pi images
            temp_dict_pi = dict() 
            for image_file in os.listdir(pi_folder):
                key_number = int(image_file[:-3])
                image_path = os.path.join(pi_folder, image_file)
                if load_memory:
                    image_path = torch.load(image_path)
                    image_path = rescale_image(image_path, output_size=resolution)
                temp_dict_pi.update({key_number:image_path})

            index_list = []
            for i in range(len(os.listdir(label_folder))):
                self.ct_file_list.append(temp_dict_ct[i])
                self.pi_file_list.append(temp_dict_pi[i])
                if self.load_labels:
                    self.label_list.append(temp_dict_json[i])
                index_list.append(len(self.ct_file_list) - 1)

            self.person_index_dict.update({person_id:index_list})
     
            if terminate:
                if person_index > terminate:
                    return None
            
        assert len(self.ct_file_list) == len(self.pi_file_list)
        #assert len(self.pi_file_list) == len(self.label_list)
            
    def __len__(self):
        return len(self.ct_file_list)

    def __getitem__(self, index):
        if self.load_labels:
            label = self.label_list[index]
        else:
            label = None
        if self.load_memory:
            ct = self.ct_file_list[index]
            #pi = self.pi_file_list[index]
        else:
            ct = torch.load(self.ct_file_list[index])
            pi = torch.load(self.pi_file_list[index])
            ct = rescale_image(ct, output_size=self.resolution)
            pi = rescale_image(pi, output_size=self.resolution)
        ct = ct.unsqueeze(0)
        pi = pi.unsqueeze(0)
        #ct_pi = torch.cat([ct, pi], dim=0)
        return ct, pi, # label


def dcm_to_dict(file_path, verbose=True):
    zd = zstd.ZstdDecompressor()
    compressed = open(file_path, "rb").read()
    try:
        if file_path[-3:] == "zst":
            data = zd.decompress(compressed)
            ds = pydicom.dcmread(BytesIO(data))
        elif file_path[-3:] == "dcm":
            ds = pydicom.dcmread(file_path)
    except:
        if verbose:
            print("error in reading", file_path)
        return None 

    arr = ds.get('pixel_array', None)#ds.pixel_array
    #print(ds["SeriesDescription"], ds["PixelSpacing"])
    if arr is None:
        if verbose:
            print("No image in", file_path)
        return None     

    slice_loc = ds.get('SliceLocation', None)
    sex = ds.get('PatientSex', None)
    weight = ds.get('PatientWeight', None)
    weight = float(weight)
    length = ds.get('PatientSize', None)
    length = float(length)
    patient_id = ds.get('PatientID', None)
    age = ds.get('PatientAge', None)
    age = int(age[:-1])
    modality = ds.get('Modality', None)
    if modality is None:
        print("No modality in:", file_path)
        return None

    series_descrip = ds.SeriesDescription.upper()
    if slice_loc is None:
        print("No slice in:", file_path)
        return None
    else:
        slice_loc = float(slice_loc)

    if modality == "CT":
        arr = apply_modality_lut(arr, ds)

    return_dict = {"slice":slice_loc, "ct":arr, "sex":sex, "weight":weight, \
                    "length":length, "id":patient_id, "age":age, "position":None, \
                    "file":file_path, "modality":modality, "series":series_descrip}
    return return_dict


def crop_resize_image(image, coeff, resolution):
    if coeff == 1.0:
        return image
    image_size = image.shape[0]
    new_size = round(image_size / coeff)
    center_x = image_size // 2 + 15
    center_y = image_size // 2 
    start_x = center_x - new_size // 2 
    start_y = center_y - new_size // 2
    end_x = center_x + new_size // 2
    end_y = center_y + new_size // 2
    croped_image = image[start_x:end_x, start_y:end_y]
    croped_image = torch.FloatTensor(croped_image).unsqueeze(0).unsqueeze(0)
    croped_image = F.interpolate(croped_image, size=(resolution,resolution) )
    croped_image = croped_image.squeeze(0).squeeze(0)
    return croped_image


# def dcm_to_dict(file_path, resolution=64, verbose=True, only_lungs=True, normalize=True):
#     zd = zstd.ZstdDecompressor()
#     compressed = open(file_path, "rb").read()
#     try:
#         if file_path[-3:] == "zst":
#             data = zd.decompress(compressed)
#             ds = pydicom.dcmread(BytesIO(data))
#         elif file_path[-3:] == "dcm":
#             ds = pydicom.dcmread(file_path)
#     except:
#         if verbose:
#             print("error in reading", file_path)
#         return None 

#     #print(ds.SeriesDescription)
#     if only_lungs and 'LUNG' not in ds.SeriesDescription.upper():
#         return None
#     #print(ds.SeriesDescription)

#     arr = ds.get('pixel_array', None)#ds.pixel_array
#     if arr is None:
#         if verbose:
#             print("No image in", file_path)
#         return None
    
#     arr = ds.pixel_array
#     hu = apply_modality_lut(arr, ds)
#     #window_center=30, window_width=150 for the liver
#     arr = normalization_window(hu, ds=None)
#     hu = torch.FloatTensor(arr)
#     if hu.max() - hu.min() == 0:
#         return None
#     if normalize:
#         hu = (hu - hu.min()) / (hu.max() - hu.min())
     
#     if resolution != 512:
#         image_tensor = hu
#         image_tensor = image_tensor.unsqueeze(0)#.unsqueeze(0)
#         image_tensor = F.interpolate(image_tensor, size=resolution)
#         image_tensor = image_tensor.permute(0, 2, 1)
#         image_tensor = F.interpolate(image_tensor, size=resolution) 
#         image_tensor = image_tensor.permute(0, 2, 1)
#         image_tensor = image_tensor.squeeze()

#     slice_loc = ds.get('SliceLocation', None)
#     sex = ds.get('PatientSex', None)
#     weight = ds.get('PatientWeight', None)
#     weight = float(weight)
#     length = ds.get('PatientSize', None)
#     length = float(length)
#     patient_id = ds.get('PatientID', None)
#     age = ds.get('PatientAge', None)
#     age = int(age[:-1])
#     modality = ds.get('Modality', None)
#     series_descrip = ds.SeriesDescription.upper()
#     if slice_loc is None:
#         print("No slice in", file_path)
#         return None
#     else:
#         slice_loc = float(slice_loc)

#     return_dict = {"slice":slice_loc, "ct":image_tensor, "sex":sex, "weight":weight, \
#                     "length":length, "id":patient_id, "age":age, "position":None, \
#                     "file":file_path, "modality":modality, "series":series_descrip}
#     return return_dict


def save_dcm_series(person_folder, dataset_path, resolution=64):
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
    #file_number = 0
    ct_list = []

    lungs_description = extract_lungs_description(path_dcm)
    if lungs_description is None:
        return None
    
    for dcm_file in os.listdir(path_dcm):
        if dcm_file[:2] == "CT" and dcm_file[-7:] == "dcm.zst":            
            file_path = os.path.join(path_dcm, dcm_file)
            ct_dict = dcm_to_dict(file_path, resolution=resolution)
            if ct_dict is None:
                continue
            else:
                #ct_dict.update({'description':lungs_description})
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

    #add pi images to ct
    pi_list = read_pi_files(path_dcm)
    if pi_list is None:
        return None

    slice_list = [x["slice"] for x in pi_list]
    for ct_element in ct_list:
        slice_location = ct_element["slice"]
        pi_interp = interpolate_image(pi_list, slice_list, slice_location)
        pi_interp = rescale_image(pi_interp, output_size=ct_element["ct"].shape[0])
        pi_interp = torch.FloatTensor(pi_interp)
        ct_element.update({"pi":pi_interp})
   
    #create folders to save files
    person_folder = os.path.join(dataset_path, ct_list[0]["id"])
    if os.path.exists(person_folder):
        rmtree(person_folder)
    os.mkdir(person_folder)
    ct_folder = os.path.join(person_folder, "ct")
    pi_folder = os.path.join(person_folder, "pi")
    label_folder = os.path.join(person_folder, "labeles")
    os.mkdir(ct_folder)
    os.mkdir(pi_folder)
    os.mkdir(label_folder)

    #label for a person containing descrition, not for an image
    person_dict = copy.copy(ct_list[0])
    del person_dict["ct"]
    del person_dict["pi"]
    del person_dict["slice"]
    del person_dict["position"]
    person_dict.update({"description":lungs_description})
    #do not forget to save here a .json !!!
    label_path = os.path.join(person_folder, "labeles.json")
    with open(label_path, 'w') as fp:
        json.dump(person_dict, fp)

    for i, ct_dict in enumerate(ct_list):
        ct_path = os.path.join(ct_folder, str(i) + ".pt")
        torch.save(ct_dict["ct"], ct_path)
        pi_path = os.path.join(pi_folder, str(i) + ".pt")
        torch.save(ct_dict["pi"], pi_path)
        del ct_dict["ct"]
        del ct_dict["pi"]
        label_path = os.path.join(label_folder, str(i) + ".json")
        with open(label_path, 'w') as fp:
            json.dump(ct_dict, fp)

    #files successfully saved
    return True


def save_dcm_series_audit(person_folder, dataset_path, resolution=64, series_type="LUNG", pt_serie="MAC"):
    sub_dir = "DICOM"
    path_dcm = os.path.join(person_folder, sub_dir)
    if not os.path.isdir(path_dcm):
        warnings.warn("the folder does not exist: "+path_dcm)
        return None
    ct_list = []
    pi_list = []
    series_list = [] #DELETEME

    description_path = os.path.join(person_folder, "заключение.txt")
    if not os.path.isfile(description_path):
        print("no description in:", person_folder)
        return None
    lungs_description = extract_lungs_from_txt(description_path)
    if lungs_description is None:
        return None
     
    for dcm_file in os.listdir(path_dcm):
        if dcm_file[-3:] == "dcm":   
            file_path = os.path.join(path_dcm, dcm_file)
            pict_dict = dcm_to_dict(file_path)
            if pict_dict is None:
                continue
            else:
                series_list.append(pict_dict["series"]) #DELETEME
                if pict_dict["modality"] == "CT" and series_type == pict_dict["series"]:
                    if pict_dict["ct"].shape[0] == 512:
                        ct_list.append(pict_dict)
                elif pict_dict["modality"] == "PT" and pt_serie == pict_dict["series"]:
                    pi_list.append(pict_dict)

    for ct in ct_list:
        window_center = -600.0
        window_width = 1500.0
        ct["ct"] = rescale_image(ct["ct"], output_size=resolution)
        ct["ct"] = normalization_window(ct["ct"], ds=None, window_center=window_center, window_width=window_width)
        ct["ct"] = (ct["ct"] - window_center + window_width / 2) / window_width
        ct["ct"] = torch.FloatTensor(ct["ct"])
    #series_list = list(set(series_list))
    #print(series_list)
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

    if pi_list is None or pi_list == []:
       print("No PI images in the study")
       return None

    for pi in pi_list:
        pi["ct"] = rescale_image(pi["ct"], output_size=resolution)
        pi["ct"] = torch.FloatTensor(pi["ct"])
    pi_list = sorted(pi_list, key=lambda x: x["slice"])
    slice_list = [x["slice"] for x in pi_list]
    for ct_element in ct_list:
        slice_location = ct_element["slice"]
        pi_interp = interpolate_image(pi_list, slice_list, slice_location)
        coeff_resize = 1.569506276016136#1.694914466286737
        pi_interp = crop_resize_image(pi_interp, coeff=coeff_resize, resolution=resolution)
        #pi_interp = rescale_image(pi_interp, output_size=ct_element["ct"].shape[0])
        #pi_interp = torch.FloatTensor(pi_interp)
        ct_element.update({"pi":pi_interp})

    #create folders to save files
    person_folder = os.path.join(dataset_path, ct_list[0]["id"])
    if os.path.exists(person_folder):
        rmtree(person_folder)
    os.mkdir(person_folder)
    ct_folder = os.path.join(person_folder, "ct")
    pi_folder = os.path.join(person_folder, "pi")
    label_folder = os.path.join(person_folder, "labeles")
    os.mkdir(ct_folder)
    os.mkdir(pi_folder)
    os.mkdir(label_folder)

    #label for a person containing descrition, not for an image
    person_dict = copy.copy(ct_list[0])
    del person_dict["ct"]
    del person_dict["pi"]
    del person_dict["slice"]
    del person_dict["position"]
    del person_dict["modality"]
    del person_dict["series"]
    person_dict.update({"description":lungs_description})
    #do not forget to save here a .json !!!
    label_path = os.path.join(person_folder, "labeles.json")
    with open(label_path, 'w') as fp:
        json.dump(person_dict, fp)

    for i, ct_dict in enumerate(ct_list):
        ct_path = os.path.join(ct_folder, str(i) + ".pt")
        torch.save(ct_dict["ct"], ct_path)
        pi_path = os.path.join(pi_folder, str(i) + ".pt")
        torch.save(ct_dict["pi"], pi_path)
        del ct_dict["ct"]
        del ct_dict["pi"]
        label_path = os.path.join(label_folder, str(i) + ".json")
        with open(label_path, 'w') as fp:
            json.dump(ct_dict, fp)

    #files successfully saved
    return True


def find_series(person_folder): 
    sub_dir = "DICOM"
    path_dcm = os.path.join(person_folder, sub_dir)
    if not os.path.isdir(path_dcm):
        warnings.warn("the folder does not exist: "+path_dcm)
        return None
    ct_series_dict = dict() 
    pt_series_dict = dict() 

    for dcm_file in os.listdir(path_dcm):
        if dcm_file[-3:] == "dcm":   
            file_path = os.path.join(path_dcm, dcm_file)
            pict_dict = dcm_to_dict(file_path)
            if pict_dict is None:
                continue
    
            serie = pict_dict["series"] 
            if pict_dict["modality"] == "CT":
                if serie in ct_series_dict.keys():
                    ct_series_dict[serie].append(pict_dict["slice"]) 
                else:
                    ct_series_dict.update({serie:[pict_dict["slice"]]})
            elif pict_dict["modality"] == "PT":
                if serie in pt_series_dict.keys():
                    pt_series_dict[serie].append(pict_dict["slice"]) 
                else:
                    pt_series_dict.update({serie:[pict_dict["slice"]]})

    for key in ct_series_dict:
        ct_series_dict[key] = sorted(ct_series_dict[key])
    for key in pt_series_dict:
        pt_series_dict[key] = sorted(pt_series_dict[key])

    return ct_series_dict
                


if __name__ == "__main__":
    #part01 for validation
    dcm_path = "/ayb/vol3/datasets/pet-ct/audit/ПЭТ-Технолоджи_Курск"
    #dcm_path = "/ayb/vol4/sftp/user23/upload/Ростов/"
    #dcm_path = "/ayb/vol3/datasets/pet-ct/part01/"
    #dcm_path = quote(dcm_path)
    dataset_path = "/ayb/vol1/kruzhilov/datasets/labeled_lungs_description/labeled_lungs_description_256/deleteme"
    print(os.path.abspath(dcm_path) )
    #print(os.path.isdir(dcm_path))
    
    resolution = 256

    if not os.path.exists(dataset_path):
        #rmtree(dataset_path)
        os.mkdir(dataset_path)
    i = 0
    for person in os.listdir(dcm_path): 
        person_folder = os.path.join(dcm_path, person)
        #print(person_folder)
        #result = save_dcm_series_audit(person_folder, dataset_path, resolution=resolution,\
        #series_type="RECON 2: CT LUNG") 
        #find_series(person_folder)
        result = save_dcm_series_audit(person_folder, dataset_path, resolution=resolution, series_type="LUNGS 1.25MM", pt_serie="3D MAC") #CTAC 5 MM THICK, MAC STANDART, LUNG 5 MM
        if result:
            print(i, person)
            i = i + 1
            break
        else:
            print(person)


    # lung_dataset = LungsLabeled(dataset_path, terminate=50, resolution=128, load_memory=False, load_labels=False)
    # #ct, pi, label = lung_dataset.__getitem__(1000)
    # key = list(lung_dataset.person_index_dict.keys())[15]
    # index = lung_dataset.person_index_dict[key][50]
    # ct = lung_dataset.__getitem__(index)
    # print(lung_dataset.person_label_dict[key])
    # plt.imshow(ct[0,:,:], cmap=plt.cm.gray)
    # plt.show()

    # print(label["slice"])
    # print("pi max", ct_pi.max())
    # plt.imshow(ct_pi[1,:,:], cmap=plt.cm.gray)
    # plt.show()

# %%
