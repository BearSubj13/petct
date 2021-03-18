import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from net import EncoderDefault
from model import Model
from dataset_with_labels import LungsLabeled
from simple_transformer import Regressor


class PersonDataset(Dataset):
    def __init__(self, dataset, person):
        super().__init__()
        self.image_file_list = []
        self.label_list = []
        self.load_memory = dataset.load_memory

        for index in dataset.person_index_dict[person]:
            image = dataset.image_file_list[index]
            self.image_file_list.append(image)
            label = dataset.label_list[index]
            self.label_list.append(label)

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, index):
        label = self.label_list[index]
        if self.load_memory:
            image = self.image_file_list[index]
        else:
            image = torch.load(self.image_file_list[index])
        return image, label


def validation(dataset, model, parameter_name='weight', device="cpu"):
    regressor.eval()
    model.eval()
    err_list = []

    for person in dataset.person_index_dict.keys():
        person_dataset = PersonDataset(dataset, person)
        person_dataloader = DataLoader(person_dataset, batch_size=10, shuffle=False, drop_last=True)
        result_sum = 0
        for image, label in person_dataloader:
            image = image.unsqueeze(1)
            image = image.to(device)
            latent = model.encoder.encode(image, lod=lod)#.squeeze(1)
            latent = latent.squeeze(1)
            predict = regressor(latent)
            predict = predict.squeeze(1)
            result_sum = result_sum + predict.sum().item()

        result = result_sum / len(person_dataset)
        label = label[parameter_name][0].item()
        err = abs(result - label)
        err_list.append(err)
    err_mean = sum(err_list) / len(err_list)
    return err_mean


#
if __name__ == "__main__":
    device = "cuda:1"
    batch_size = 30
    parameter_name='weight'
    dataset_path = "/ayb/vol1/kruzhilov/datasets/labeled_lungs/train"
    weights_path = "/home/kruzhilov/petct/weights/model64_5layers.pth"

    lung_dataset = LungsLabeled(dataset_path, terminate=20, load_memory=True)
    train_val = train_val_split(lung_dataset, val_split=0.2)
    lung_dataset_train = train_val['train']
    lung_dataset_val = train_val['val']
    print('train dataset size:', len(lung_dataset_train))
    print('val dataset size:', len(lung_dataset_val))
    # encoder = EncoderDefault(startf=32, maxf=256, latent_size=128, layer_count=5, \
    #           channels=1)
    model = Model(channels=1, device=device, layer_count=5, latent_size=128)
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    regressor = Regressor(d_model=128)
    regressor.to(device)            
    lod = 4 

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': regressor.parameters()}], lr=0.0002, weight_decay=0.001)
    lungs_dataloader = DataLoader(lung_dataset_train, batch_size=32, shuffle=True)
    loss_f = nn.MSELoss()

    for epoch in range(100):
        loss_list = []
        regressor.train()
        model.train()

        for image, label in lungs_dataloader:
            label = label['weight'].type(torch.FloatTensor)
            label = label.to(device)
            image = image.unsqueeze(1)
            image = image.to(device)
            latent = model.encoder.encode(image, lod=lod)
            latent = latent.squeeze(1)
            predict = regressor(latent)
            predict = predict.squeeze(1)
            assert predict.shape == label.shape
            loss = loss_f(predict, label)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
        train_err = sum(loss_list) / len(loss_list)
        val_err = validation(lung_dataset_val, model, parameter_name='weight', device=device)
        print(epoch, 'train:', train_err, 'val:', val_err)
