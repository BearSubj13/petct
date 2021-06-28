#%%
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model import Model
from luna16 import train_val_split_luna, Luna_heat_map_dataset
from utils.decoder import Decoder1


if __name__ == "__main__":
    number_of_blocks = 6
    resolution_power = 7    
    load_model_path = 'weights/weights_ct256/model128_6layers_steps30plus.pth'
    load_decoder_path = 'weights/luna_heatmap/decoder_freeze.pth'
    #save_model_path = 'weights/luna_heatmap/heatmap128_6layers.pth'
    #save_decoder_path = 'weights/luna_heatmap/decoder.pth'
    weight_decay = 0.0001
    batch_size = 54
    lr = 0.0001
    device = "cuda:3"
    dataset_path = "/ayb/vol1/kruzhilov/datasets/luna16_heatmap/resolution128/"

    train_dataset, val_dataset = train_val_split_luna(dataset_path, val_rate=0.1)
    print("train len:", len(train_dataset), ", val len:", len(val_dataset))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Model(channels=1, device=device, layer_count=number_of_blocks, latent_size=128)
    decoder = Decoder1()
    model.load_state_dict(torch.load(load_model_path, map_location="cpu"))
    decoder.load_state_dict(torch.load(load_decoder_path, map_location="cpu"))
    image, heatmap = val_dataset.__getitem__(16)
    image_tensor = torch.FloatTensor(image.copy())
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    lod = resolution_power - 2
    styles = model.encode(image_tensor, lod, blend_factor=1)[0].squeeze(1)
    #s = styles.view(styles.shape[0], 1, styles.shape[1])
    #styles = s.repeat(1, model.mapping_f.num_layers, 1)

    image_reconstructed = decoder(styles)
    image_reconstructed = torch.sigmoid(image_reconstructed)
    image_reconstructed = image_reconstructed.detach().numpy()
    image_reconstructed = image_reconstructed.squeeze(0).squeeze(0)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()
    plt.imshow(image_reconstructed, cmap=plt.cm.gray)
    plt.show()
    plt.imshow(heatmap, cmap=plt.cm.gray)
    plt.show()
    print(" ") 

# %%
