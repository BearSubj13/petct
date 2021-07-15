#%%
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
import skimage

#from model import Model
from unet.unet_model import UNet
from luna16 import train_val_split_luna, find_stars, coordinates_metrics
from utils.decoder import Decoder1


if __name__ == "__main__":
    number_of_blocks = 6
    resolution_power = 8    
    load_model_path = 'weights/luna_heatmap/unet256_pos1_8.pth'
    #load_decoder_path = 'weights/luna_heatmap/decoder_unfreeze.pth'
    device = "cuda:3"
    dataset_path = "/ayb/vol1/kruzhilov/datasets/luna16_heatmap/resolution256_largeheatmap/"
    item_number = 95
    #succeded 20, 30, 50, 55, 60, 75, 80, 85, 90, 95(very precise)
    #succeded 160(big), 145, 140(big), 100(vhp), 115(very big), 125(vhp), 130, 137(the same leison than 135)
    #one succeded, one false positive 105, 
    #failed 40, 66-70, 110, 135, 150, 152-155, 

    train_dataset, val_dataset = train_val_split_luna(dataset_path, val_rate=0.1)
    print("train len:", len(train_dataset), ", val len:", len(val_dataset))
    #train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    unet = UNet(n_channels=1, n_classes=1)
    unet.load_state_dict(torch.load(load_model_path, map_location="cpu"))
    
    image, heatmap, coordinates = val_dataset.__getitem__(item_number)
    print("item number: ", item_number)
    coordinates = coordinates.reshape([-1, 3])
    image_tensor = torch.FloatTensor(image.copy())
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    image_reconstructed = unet(image_tensor)
    # lod = resolution_power - 2
    # styles = model.encode(image_tensor, lod, blend_factor=1)[0].squeeze(1)
    # s = styles.view(styles.shape[0], 1, styles.shape[1])
    # styles = s.repeat(1, model.mapping_f.num_layers, 1)

    # image_reconstructed = decoder(styles)
    # image_reconstructed = torch.sigmoid(image_reconstructed)
    image_reconstructed = image_reconstructed.detach().numpy()
    image_reconstructed = image_reconstructed.squeeze(0).squeeze(0)

    xyr = find_stars(image_reconstructed)
   
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()

    fig, ax = plt.subplots()
    plt.imshow(image, cmap=plt.cm.gray)
    for i in range(coordinates.shape[0]):
        x = coordinates[i, 0]
        y = coordinates[i, 1]
        r = coordinates[i, 2]
        circle = plt.Circle((x, y), 1.2*r, color='r', fill=False)
        ax.add_artist(circle)
    for i in range(xyr.shape[0]):
        x = xyr[i, 0]
        y = xyr[i, 1]
        r = xyr[i, 2]
        circle = plt.Circle((x, y), 1.2*r, color='g', fill=False)
        ax.add_artist(circle)
    plt.show()
    
    fig, ax = plt.subplots()
    plt.imshow(image_reconstructed, cmap=plt.cm.gray)
    for i in range(coordinates.shape[0]):
        x = coordinates[i, 0]
        y = coordinates[i, 1]
        r = coordinates[i, 2]
        circle = plt.Circle((x, y), 1.2*r, color='r', fill=False)
        ax.add_artist(circle)
    for i in range(xyr.shape[0]):
        x = xyr[i, 0]
        y = xyr[i, 1]
        r = xyr[i, 2]
        circle = plt.Circle((x, y), 1.2*r, color='g', fill=False)
        ax.add_artist(circle)
    plt.show()

    fig, ax = plt.subplots()
    plt.imshow(heatmap, cmap=plt.cm.gray)
    for i in range(xyr.shape[0]):
        x = xyr[i, 0]
        y = xyr[i, 1]
        r = xyr[i, 2]
        circle = plt.Circle((x, y), 1.2*r, color='g', fill=False)
        ax.add_artist(circle)
    plt.show()
    print(" ") 


    prec, recall, std_coord = coordinates_metrics(xyr, coordinates, efficient_radius=2.0)
    print(prec, recall, std_coord)
# %%
