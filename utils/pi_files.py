#%%
import os
from shutil import rmtree
import json
from tqdm import tqdm
import bisect
import warnings

import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_file
from pydicom.pixel_data_handlers.util import apply_modality_lut
import zstandard as zstd
from io import BytesIO


def find_le(a, x):
    'Find rightmost value less than or equal to x'
    i = bisect.bisect_right(a, x)
    if i:
        return i - 1
    raise ValueError

def find_ge(a, x):
    'Find leftmost item greater than or equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return i
    raise ValueError


def read_pi_files(person_folder, verbose=True):
    """
    person_folder: str - path to dcm files
    return: sorted by slice value list of {"slice":, "image":} pi images
    """
    # dir_list = os.listdir(person_folder)
    # if len(dir_list) == 1:
    #     sub_dir = dir_list[0]
    # elif len(dir_list) > 1:
    #     for sub_dir in dir_list:
    #         #looks for an non-empty sub_dir
    #         full_path = os.path.join(person_folder, sub_dir)
    #         if len(os.listdir(full_path)) > 0:
    #             break  
    # else:
    #     return None 
    path_dcm = person_folder #os.path.join(person_folder, sub_dir)


    zd = zstd.ZstdDecompressor()
    pi_list = []
    for i, file_name in enumerate(os.listdir(path_dcm)):
        if file_name[:2] == "PI" and file_name[-7:] == "dcm.zst":
            file_path = os.path.join(path_dcm, file_name)
            compressed = open(file_path, "rb").read()
        else:
            continue

        try:
            data = zd.decompress(compressed)
            ds = pydicom.dcmread(BytesIO(data))
        except:
            if verbose:
                print("error in reading", file_path)
            return None 

        arr = ds.get('pixel_array', None) 
        if arr.shape[0] != 256:
            continue     
        slice_loc = ds.get('SliceLocation', None)
        
        if slice_loc is None:
            print("No slice in", file_path)
            continue        
        else:
            slice_loc = float(slice_loc)

        pi_dict = {'slice':slice_loc, "image":arr}#, "normalized":normalized_image}
        pi_list.append(pi_dict)

    if len(pi_list) == 0:
        return None

    pi_list = sorted(pi_list, key=lambda x: x["slice"])

    if pi_list[0]['slice'] == pi_list[1]['slice']:
        pi_list = pi_list[1::2]
        
    return pi_list


def rescale_image(image,  output_size=128):
    """
    an elegant way to resize image in numpy in a pooling way
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


def interpolate_image(pi_list, slice_list, slice_interp):
    if slice_interp < pi_list[0]["slice"]:
        return pi_list[0]["image"]    
    if slice_interp > pi_list[-1]["slice"]:
        return pi_list[-1]["image"]
    #assert slice_interp >= pi_list[0]["slice"]
    #assert slice_interp <= pi_list[-1]["slice"]
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
    person_folder = "/ayb/vol3/datasets/pet-ct/part01/990003766"
    pi_list = read_pi_files(person_folder)
    slice_list = [x["slice"] for x in pi_list]

    k = 30
    image0 = pi_list[k]["image"]
    t0 = pi_list[k]["slice"]
    image1 = pi_list[k + 1]["image"]
    t1 = pi_list[k + 1]["slice"]
    #an elegant way to resize image in numpy in a pooling way
    image2 = pi_list[k + 2]["image"]
    t2 = pi_list[k + 2]["slice"]

    tinterp = -950
    #d = t2 - t0
    image_interp = interpolate_image(pi_list, slice_list, tinterp)
    #image0 + (tinterp - t0) * (image2 - image0) / d 
    
    print('slice 0:', t0)
    plt.imshow(image0, cmap=plt.cm.gray)
    plt.show()
    print('slice interpolate:', tinterp)
    plt.imshow(image_interp, cmap=plt.cm.gray)
    plt.show()
    print('slice 1:', t1)
    plt.imshow(image1, cmap=plt.cm.gray)
    plt.show()
    print('slice 2:', t2)
    plt.imshow(image2, cmap=plt.cm.gray)
    plt.show()
# %%
