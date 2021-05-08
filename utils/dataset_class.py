#%%
import os
from shutil import rmtree
import json
from tqdm import tqdm
import warnings
import bisect

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class LungsLabeled(Dataset):
    """
    dataset_path: string - path to the dataset
    terminate: int or False - stop add new values to dataset if index > terminate
    load_memory: boolean - if True loads images to memory
    """
    def __init__(self, dataset_path, terminate=False, load_memory=False, load_slices_labels=True):
        super().__init__()
        self.load_slices_labels = load_slices_labels
        self.load_memory = load_memory
        #files indexes for a person in the list - "person id":[list of indexes]
        self.person_label_dict = dict()
        #a dictionary contains all information about a person e.g. weight, length, sex, ct description
        self.person_index_dict = dict()
        self.ct_file_list = []
        self.pi_file_list = []
        self.label_list = []

        for person_index, person_id in enumerate(tqdm(os.listdir(dataset_path))):
            person_folder = os.path.join(dataset_path, person_id)
            label_folder = os.path.join(person_folder, 'labeles')
            ct_folder = os.path.join(person_folder, 'ct')
            pi_folder = os.path.join(person_folder, 'pi')
            #pi_folder = os.path.join(person_folder, 'pi')
            assert len(os.listdir(label_folder)) == len(os.listdir(ct_folder))
            assert len(os.listdir(label_folder)) == len(os.listdir(pi_folder))

            #a label for a person with description
            label_path = os.path.join(person_folder, "labeles.json")
            with open(label_path, 'r') as fp:
                person_label = json.load(fp)
            self.person_label_dict.update({person_id:person_label})

            #add labels
            if self.load_slices_labels:
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

            #add ct images
            temp_dict_ct = dict() 
            for image_file in os.listdir(ct_folder):
                key_number = int(image_file[:-3])
                image_path = os.path.join(ct_folder, image_file)
                if load_memory:
                    image_path = torch.load(image_path)
                temp_dict_ct.update({key_number:image_path})

            #add pi images
            temp_dict_pi = dict() 
            for image_file in os.listdir(pi_folder):
                key_number = int(image_file[:-3])
                image_path = os.path.join(pi_folder, image_file)
                if load_memory:
                    image_path = torch.load(image_path)
                temp_dict_pi.update({key_number:image_path})

            index_list = []
            for i in range(len(os.listdir(label_folder))):
                self.ct_file_list.append(temp_dict_ct[i])
                self.pi_file_list.append(temp_dict_pi[i])
                if self.load_slices_labels: 
                    self.label_list.append(temp_dict_json[i])
                index_list.append(len(self.ct_file_list) - 1)

            self.person_index_dict.update({person_id:index_list})
     
            #early termination of loading data - for debug only
            if terminate:
                if person_index > terminate:
                    return None
        if not self.load_slices_labels: return 
        assert len(self.ct_file_list) == len(self.label_list)
        assert len(self.pi_file_list) == len(self.label_list)
            
    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, index):
        if self.load_slices_labels:
            label = self.label_list[index]
        else:
            label = ''
        if self.load_memory:
            ct = self.ct_file_list[index]
            pi = self.pi_file_list[index]
        else:
            ct = torch.load(self.ct_file_list[index])
            pi = torch.load(self.pi_file_list[index])
        return ct, pi, label


def rescale_image(image,  output_size=128):
    """
    an elegant way to resize an image in numpy in a pooling way
    image: numpy - should have a size of output_size*n
    output_size: int - a size for interpolation
    """
    input_size = image.shape[0]
    if input_size == output_size:
        return image
    else:
        bin_size = input_size // output_size
        small_image = image.reshape((1, output_size, bin_size,
                                      output_size, bin_size))
        small_image = small_image.mean(4).mean(2)
        small_image = small_image.squeeze(0)
        return small_image


def find_le(a, x):
    'Finds rightmost value less than or equal to x in a list a'
    """
    Find rightmost value less than or equal to x in a list a
    a: list
    x: float
    return: int - element position in a list
    """
    i = bisect.bisect_right(a, x)
    if i:
        return i - 1
    raise ValueError


def find_ge(a, x):
    """
    Find leftmost item greater than or equal to x in a list a
    a: list
    x: float
    return: int - element position in a list
    """
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return i
    raise ValueError


def interpolate_image(pi_list, slice_list, slice_interp):
    """
    pi_list: list of numpy - pi images
    slice_list: list of float - screening position in mm
    slice_interp: float - position to interpolate an image in
    """
    if slice_interp < pi_list[0]["slice"]:
        return pi_list[0]["image"]    
    if slice_interp > pi_list[-1]["slice"]:
        return pi_list[-1]["image"]

    i_left = find_le(slice_list, slice_interp)
    i_right = find_ge(slice_list, slice_interp)
    if i_left == i_right:
        return pi_list[i_left]["image"]
    
    slice_left = pi_list[i_left]["slice"]
    slice_right = pi_list[i_right]["slice"]
    image_left = pi_list[i_left]["image"]
    image_right = pi_list[i_right]["image"]

    d = slice_right - slice_left
    if d != 0:
        image_interp = image_left + (slice_interp - slice_left) * (image_right - image_left) / d 
    else:
        image_interp = image_left
        warnings.warn("Left and right images are equal in interpolation!")
    return image_interp

 
if __name__ == "__main__":
    dcm_path = "/ayb/vol3/datasets/pet-ct/part0/"
    dataset_path = "/ayb/vol1/kruzhilov/datasets/labeled_lungs_description/train"

    lung_dataset = LungsLabeled(dataset_path, terminate=5, load_memory=False)
    #ct, pi, label = lung_dataset.__getitem__(1000)
    key = list(lung_dataset.person_index_dict.keys())[3] #third person ID
    index = lung_dataset.person_index_dict[key][50] #50th image of the 3d person
    
    print("id:", key)
    print(lung_dataset.person_label_dict[key])
    
    ct, pi, label = lung_dataset.__getitem__(index)

    plt.imshow(ct, cmap=plt.cm.gray)
    plt.show()
    print(label["slice"], "mm")
    plt.imshow(pi, cmap=plt.cm.gray)
    plt.show()

    #slices = [lung_dataset.label_list[index]["slice"] for index in lung_dataset.person_index_dict[key]]
    #print(slices)


# %%
