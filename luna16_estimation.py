#%%

import os
#from PIL.Image import blend
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
#from albumentations import VerticalFlip #as alb

from model import Model
from luna16 import train_val_split_luna, Luna_heat_map_dataset
from utils.decoder import Decoder1


def train_epoch(model, decoder, data_loader, encoder_optimizer, decoder_optimizer, device, lod, freeze_encoder=False):
    model.train()
    decoder.train()
    pos_weigth = torch.FloatTensor([2.0])
    pos_weigth = pos_weigth.to(device)
    loss_f = nn.BCEWithLogitsLoss(pos_weight=pos_weigth)
    loss_list = []
    for image, heat_map in data_loader:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        image = image.type(torch.FloatTensor)
        image = image.unsqueeze(1)
        image = image.to(device)
        heat_map = heat_map.to(device)
        if freeze_encoder:
            with torch.no_grad():
                styles = model.encode(image, lod, blend_factor=1)[0].squeeze(1)
        else:
            styles = model.encode(image, lod, blend_factor=1)[0].squeeze(1)
        # s = styles.view(styles.shape[0], 1, styles.shape[1])
        # styles = s.repeat(1, model.mapping_f.num_layers, 1)
        # image_reconstructed = model.decoder(styles, lod=lod, blend=1, noise=False)
        # assert image_reconstructed.shape == image.shape
        # image_reconstructed = image_reconstructed.squeeze(1)
        image_reconstructed = decoder(styles)
        image_reconstructed = image_reconstructed.squeeze(1)
        assert image_reconstructed.shape == heat_map.shape

        loss = loss_f(image_reconstructed, heat_map)
        loss_list.append(loss.item())
        loss.backward()
        if not freeze_encoder:
            encoder_optimizer.step()
        decoder_optimizer.step()
    mean_loss = sum(loss_list) / len(loss_list)
    return mean_loss


def eval_epoch(model, decoder, data_loader, device, lod):
    model.eval()
    decoder.eval()
    loss_f = nn.BCEWithLogitsLoss()
    loss_list = []
    for image, heat_map in data_loader:
        image = image.type(torch.FloatTensor)
        image = image.unsqueeze(1)
        image = image.to(device)
        heat_map = heat_map.to(device)
        with torch.no_grad():
            styles = model.encode(image, lod, blend_factor=1)[0].squeeze(1)
            # s = styles.view(styles.shape[0], 1, styles.shape[1])
            # styles = s.repeat(1, model.mapping_f.num_layers, 1)
            # image_reconstructed = model.decoder(styles, lod=lod, blend=1, noise=False)
            # assert image_reconstructed.shape == image.shape
            # image_reconstructed = image_reconstructed.squeeze(1)
            image_reconstructed = decoder(styles)
            image_reconstructed = image_reconstructed.squeeze(1)
            assert image_reconstructed.shape == heat_map.shape
        loss = loss_f(image_reconstructed, heat_map)
        loss_list.append(loss.item())
    mean_loss = sum(loss_list) / len(loss_list)
    return mean_loss



if __name__ == "__main__":
    number_of_blocks = 6
    resolution_power = 7
    load_model_path = 'weights/weights_ct256/model128_6layers_steps30plus.pth'
    load_decoder_path = None#'weights/luna_heatmap/decoder.pth'
    save_model_path = None#'weights/luna_heatmap/heatmap128_6layers_freeze.pth'
    save_decoder_path = 'weights/luna_heatmap/decoder_freeze.pth'
    weight_decay = 0.0001
    batch_size = 50
    lr = 0.0001
    freeze_encoder = True
    device = "cuda:3"
    dataset_path = "/ayb/vol1/kruzhilov/datasets/luna16_heatmap/resolution128/"

    model = Model(channels=1, device=device, layer_count=number_of_blocks, latent_size=128)
    decoder = Decoder1()
    model = model.to(device)
    decoder = decoder.to(device)
    if load_model_path:
        model.load_state_dict(torch.load(load_model_path, map_location=device))
    if load_decoder_path:
        decoder.load_state_dict(torch.load(load_decoder_path, map_location=device))

    train_dataset, val_dataset = train_val_split_luna(dataset_path, val_rate=0.1)
    print("train len:", len(train_dataset), ", val len:", len(val_dataset))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    encoder = model.encoder
    decoder_optimizer = torch.optim.Adam([
        {'params': decoder.parameters()}
    ], lr=lr, weight_decay=0)

    encoder_optimizer = torch.optim.Adam([
        {'params': encoder.parameters()},
    ], lr=lr, weight_decay=weight_decay)

    lod = resolution_power - 2
    for epoch in range(10000):    
        loss_train = train_epoch(model, decoder, train_data_loader, encoder_optimizer, decoder_optimizer, device, lod, freeze_encoder)
        loss_val = eval_epoch(model, decoder, train_data_loader, device, lod)
        print(epoch, loss_train, loss_val)
        if (epoch >= 10) and ((epoch % 10) == 0):
            if save_model_path:
                torch.save(model.state_dict(), save_model_path)
            torch.save(decoder.state_dict(), save_decoder_path)
    
    if save_model_path:
        torch.save(model.state_dict(), save_model_path)
    torch.save(decoder.state_dict(), save_decoder_path)


# %%
