import os
import torch
from torch import nn
from torch.nn import functional as F

from net import EncoderDefault
from dataset_with_labels import LungsLabeled
from simple_transformer import Regressor
from torch.utils.data import DataLoader

#
if __name__ == "__main__":
    dataset_path = "/ayb/vol1/kruzhilov/datasets/labeled_lungs/train"
    lung_dataset = LungsLabeled(dataset_path, terminate=True)
    encoder = EncoderDefault(startf=32, maxf=256, latent_size=128, layer_count=5, \
              channels=1)
    regressor = Regressor(d_model=128)
    image, label = lung_dataset.__getitem__(0) 
    image = image.unsqueeze(0).unsqueeze(0)            
    lod = 4
    # encoder.eval()
    # latent = encoder.encode(image, lod=lod)
    # latent = latent.squeeze(1)
    # regressor.eval()
    # result = regressor(latent)

    optimizer = torch.optim.Adam([
        {'params': encoder.parameters()},
        {'params': regressor.parameters()}], lr=0.01, weight_decay=0.0001)
    lungs_dataloader = DataLoader(lung_dataset, batch_size=20, shuffle=True)
    loss_f = nn.MSELoss()

    for epoch in range(2):
        for image, label in lungs_dataloader:
            label = label['weight'].type(torch.FloatTensor)
            image = image.unsqueeze(1)
            latent = encoder.encode(image, lod=lod)
            latent = latent.squeeze(1)
            predict = regressor(latent)
            predict = predict.squeeze(1)
            assert predict.shape == label.shape
            loss = loss_f(predict, label)
            print(loss.item())
            loss.backward()
            optimizer.step()

