
#%%
import os
import pydicom
from pydicom.data import get_testdata_file
from pydicom.pixel_data_handlers.util import apply_modality_lut
import zstandard as zstd
from io import BytesIO
import matplotlib.pyplot as plt
import torch as tr


def lung_files(folder_path):
    zd = zstd.ZstdDecompressor()
    files_list = []

    for file_name in os.listdir(folder_path):
        if '.dcm' in file_name:
           fname = os.path.join(folder_path, file_name)
           compressed = open(fname, "rb").read()
           data = zd.decompress(compressed)
           ds = pydicom.dcmread(BytesIO(data))
           if 'Lung' in ds.SeriesDescription:
               files_list.append(file_name)
    return files_list


def normalization_window(image, ds):
    if "WindowCenter" in ds and "WindowWidth" in ds:
        window_center = ds["WindowCenter"].value
        window_width = ds["WindowWidth"].value
        #print(window_center, window_width)

        image_min = window_center - window_width / 2
        image_max = window_center + window_width / 2

        image[image < image_min] = image_min
        image[image > image_max] = image_max

        return image


def dcm_to_image(file_path):
    zd = zstd.ZstdDecompressor()
    compressed = open(file_path, "rb").read()
    data = zd.decompress(compressed)
    ds = pydicom.dcmread(BytesIO(data))

    arr = ds.pixel_array
    hu = apply_modality_lut(arr, ds)
    arr = normalization_window(hu, ds)
    hu = tr.FloatTensor(hu)
    hu = (hu - hu.min()) / (hu.max() - hu.min())
    return hu


def add_files_to_dataset(dcm_files_path, dataset_path):
    lung_files_list = lung_files(dcm_files_path)
    for file_name in lung_files_list:
        print(file_name)
        file_path = os.path.join(path_dcm, lung_files_list[0])
        hu_tensor = dcm_to_image(file_path)
        name_without_extension = file_name[:-4]
        file_name_in_dataset = name_without_extension + '.pt'
        file_path_in_dataset = os.path.join(dataset_path, file_name_in_dataset)
        result_dict = hu_tensor
        tr.save(result_dict, file_path_in_dataset)




# path_dcm = "/ayb/vol3/datasets/pet-ct/part01/20000060/70004030"

# zd = zstd.ZstdDecompressor(path_dcm)

# fname = "CT.1.2.840.113619.2.290.3.279707939.199.1523331541.887.3.dcm.zst"
# #fname = "CT.1.2.840.113619.2.290.3.279707939.199.1523331542.286.780.dcm.zst"
# fname = "CT.1.2.840.113619.2.290.3.279707939.199.1523331542.286.571.dcm.zst"    

# path_dcm = "/ayb/vol3/datasets/pet-ct/part01/70000981/70004014"
# fname = "CT.1.2.840.113619.2.290.3.279707939.213.1522899942.398.10.dcm.zst"

# fname = os.path.join(path_dcm, fname)

# compressed = open(fname, "rb").read()
# data = zd.decompress(compressed)
# ds = pydicom.dcmread(BytesIO(data))

# #print(ds)

# # Normal mode:
# print()
# print(f"File path........: {fname}")
# print(f"SOP Class........: {ds.SOPClassUID} ({ds.SOPClassUID.name})")
# print(f"Series descriptgion: {ds.SeriesDescription}")
# print()

# pat_name = ds.PatientName
# #display_name = pat_name.family_name + ", " + pat_name.given_name
# #print(f"Patient's Name...: {display_name}")
# print(f"Patient ID.......: {ds.PatientID}")
# print(f"Modality.........: {ds.Modality}")
# print(f"Study Date.......: {ds.StudyDate}")
# print(f"Image size.......: {ds.Rows} x {ds.Columns}")
# print(f"Pixel Spacing....: {ds.PixelSpacing}")

# # use .get() if not sure the item exists, and want a default value if missing
# print(f"Slice location...: {ds.get('SliceLocation', '(missing)')}")

# # plot the image using matplotlib
# plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
# plt.show()

if __name__ == "__main__":
    #path_dcm = "/ayb/vol3/datasets/pet-ct/part01/70000981/70004014"
    #path_dcm = "/ayb/vol3/datasets/pet-ct/part01/20000060/70004030"
    #path_dcm = "/ayb/vol3/datasets/pet-ct/part01/70001008/70004169"
    #path_dcm = "/ayb/vol3/datasets/pet-ct/part01/990002892/70004198"
    path_dcm = "/ayb/vol3/datasets/pet-ct/part01/990004814/70004155"
    dataset_path = "/ayb/vol1/kruzhilov/lungs_images/"
    add_files_to_dataset(path_dcm, dataset_path)

    # lung_files_list = lung_files(path_dcm)
    # file_path = os.path.join(path_dcm, lung_files_list[0])
    # print(file_path)
    # hu_tensor = dcm_to_image(file_path)
    # hu_numpy = hu_tensor.detach().numpy()
    # plt.imshow(hu_numpy, cmap=plt.cm.gray)
    # plt.show()


# %%
