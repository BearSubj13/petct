import os
from tqdm import tqdm
from time import sleep

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
#from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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


def validation(dataset, model, parameter_name, device="cpu"):
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
        err = (result - label)**2
        err_list.append(err)
    err_mean = np.sqrt( sum(err_list) / len(err_list) )
    return err_mean


def validation_classification(dataset, model, parameter_name, device="cpu"):
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
            result_sum = result_sum + torch.sigmoid(predict).sum().item()

        result = result_sum / len(person_dataset)
        result = round(result) #round to 0 or 1
        label = label[parameter_name][0].item()
        err = (result, label)
        err_list.append(err)
    y_pred, y_true = zip(*err_list)
    report = classification_report(y_true, y_pred, output_dict=True)
    return report["accuracy"], report


def statistics_dataset(dataset, parameter_name, mean_train=None):
     value_list = []
     for person in dataset.person_index_dict.keys():
         index = dataset.person_index_dict[person][0]
         value = dataset.label_list[index][parameter_name]
         value_list.append(value)
     value_list = np.array(value_list)
     mean_value = value_list.mean()
     if mean_train is None:
        std_value = value_list.std()
     else:
        value_list = value_list - mean_train 
        value_list = value_list**2
        std_value = np.sqrt(value_list.mean())
     return mean_value, std_value


if __name__ == "__main__":
    device = "cuda:1"
    batch_size = 20
    parameter_name='sex'
    freeze_encoder = False
    classification = True
    dataset_path_train = "/ayb/vol1/kruzhilov/datasets/labeled_lungs/train"
    dataset_path_val = "/ayb/vol1/kruzhilov/datasets/labeled_lungs/val"
    weights_path = "/home/kruzhilov/petct/weights/model64_5layers.pth"

    lung_dataset_train = LungsLabeled(dataset_path_train, terminate=500, load_memory=True)
    lung_dataset_val = LungsLabeled(dataset_path_val, terminate=50, load_memory=True)

    print('')
    print("parameter name:", parameter_name)
    print("batch size:", batch_size)
    print("freeze encoder:", freeze_encoder)
    number_of_persons = len(lung_dataset_train.person_index_dict)
    print('train dataset size:', len(lung_dataset_train), ', number of persons:', number_of_persons, flush=True)
    number_of_persons = len(lung_dataset_val.person_index_dict)
    print('val dataset size:', len(lung_dataset_val), ', number of persons:', number_of_persons, flush=True)

    mean_train, std_train = statistics_dataset(lung_dataset_train, parameter_name)
    mean_val, std_val = statistics_dataset(lung_dataset_val, parameter_name)
    _, std_train_val = statistics_dataset(lung_dataset_val, parameter_name, mean_train=mean_train)
    print("mean train:{0:3.3f}, std train:{1:2.3f}".format(mean_train, std_train))
    print("mean val:{0:3.3f},   std val:{1:2.3f}".format(mean_val, std_val))
    print("                 std train val:{0:2.3f}".format(std_train_val))

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
        {'params': regressor.parameters()}], lr=0.0001, weight_decay=0.001)
    lungs_dataloader_train = DataLoader(lung_dataset_train, batch_size=32, shuffle=True)
    #loss_f = nn.MSELoss()
    loss_f = nn.BCEWithLogitsLoss()

    optimum_val_err = None
    for epoch in range(150):
        loss_list = []
        regressor.train()
        model.train()

        for image, label in tqdm(lungs_dataloader_train):
            label = label[parameter_name].type(torch.FloatTensor)
            label = label.to(device)
            image = image.unsqueeze(1)
            image = image.to(device)
            if freeze_encoder:
                with torch.no_grad():
                    latent = model.encoder.encode(image, lod=lod)
            else:
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
        if classification:
            val_err, report = validation_classification(lung_dataset_val, model, parameter_name=parameter_name, device=device)
        else:
            val_err = validation(lung_dataset_val, model, parameter_name=parameter_name, device=device)

        if optimum_val_err is None:
            optimum_val_err = val_err

        if classification:
            if val_err > optimum_val_err:
                optimum_val_err = val_err
                optim_report = report
            else:    
                min_val_err = min(val_err, optimum_val_err)
        print('', flush=True)
        print(epoch, 'train:', train_err, 'val:', val_err, flush=True)
    
    if classification:
        print('accuracy:', optimum_val_err)
        print(optim_report)
    else:
        print('min validation std:{0:2.3f}'.format(min_val_err))
        R2 = 1 - min_val_err**2 / std_train_val**2
        print("R2={0:2.3f}".format(R2))
