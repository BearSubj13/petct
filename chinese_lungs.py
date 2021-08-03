#%%
import os
import xml.etree.ElementTree as et

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from create_dataset import normalization_window

import torch
from unet.unet_model import UNet
from luna16 import find_stars
from utils.pi_files import rescale_image


if __name__ == "__main__":
    load_model_path = 'weights/luna_heatmap/unet256_pos1_8.pth'
    unet = UNet(n_channels=1, n_classes=1)
    unet.load_state_dict(torch.load(load_model_path, map_location="cpu"))

    folder_path = "/ayb/vol1/kruzhilov/datasets/chineselungs/"
    person_path = "manifest-1608669183333/Lung-PET-CT-Dx/Lung_Dx-A0002/"
    file_path = "04-25-2007-ThoraxAThoraxRoutine Adult-34834/3.000000-ThoraxRoutine  8.0.0  B40f-10983"
    #dcm_file = "1-09.dcm"
    path = os.path.join(folder_path, person_path)
    path_dcm_folder = os.path.join(path, file_path)
    #dcm_path = os.path.join(path, dcm_file)

    labeled_frames_list = []
    annotation_path = "Annotation/A0002/"
    path = os.path.join(folder_path, annotation_path)
    for file_name in os.listdir(path):
        labeled_frames_list.append(file_name[:-4])

    for dcm_file in os.listdir(path_dcm_folder):
        dcm_path = os.path.join(path_dcm_folder, dcm_file)
        ds = pydicom.dcmread(dcm_path)
        uid = ds.get("SOPInstanceUID") #"FrameOfReferenceUID")        
        if uid in labeled_frames_list: 
            annotation_file = uid + ".xml"
    
            image = ds.pixel_array
            hu = apply_modality_lut(image, ds)
            image = normalization_window(hu, None)
            
            image = rescale_image(image, output_size=256)
            image = (image - image.min()) / (image.max() - image.min())
            image_tensor = torch.FloatTensor(image.copy())
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
            image_reconstructed = unet(image_tensor)
            image_reconstructed = image_reconstructed.detach().numpy()
            image_reconstructed = image_reconstructed.squeeze(0).squeeze(0)
            xyr = find_stars(image_reconstructed)

            path = os.path.join(folder_path, annotation_path)
            annotation_file_path = os.path.join(path, annotation_file)
            tree = et.parse(annotation_file_path)
            root = tree.getroot()#.get("object")
            bndbox = []
            for child in root.find("object").find("bndbox"):
                #print(child.tag, child.text)
                bndbox.append(int(child.text))
            xmin = bndbox[0] // 2
            ymin = bndbox[1] // 2
            xmax = bndbox[2] // 2
            ymax = bndbox[3] // 2
            dx = xmax - xmin
            dy = ymax - ymin

            fig, ax = plt.subplots()
            plt.axis('off') 
            plt.imshow(image, cmap=plt.cm.gray)
            rect = patches.Rectangle((xmin, ymin), dx, dy, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_artist(rect)
            for i in range(xyr.shape[0]):
                x = xyr[i, 0]
                y = xyr[i, 1]
                r = xyr[i, 2]
                circle = plt.Circle((x, y), 1.2*r, color='g', fill=False)
                ax.add_patch(circle)
            plt.show()


# %%
