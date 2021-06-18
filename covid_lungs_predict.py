#%%
#prediction result by frozen encoder
#accuracy 93%, ROC AUC 98%
import numpy as np
import os
import matplotlib.pyplot as plt
import tqdm
from copy import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, SubsetRandomSampler
from torch.nn import functional as F
#from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from utils.pi_files import rescale_image
from model import Model

class LungsCOVID(Dataset):
    """
    """
    def __init__(self, dataset_path, resolution=64, terminate=False, load_memory=True):
        super().__init__()
        self.load_memory = load_memory
        self.resolution = resolution
        #files indexes for a person in the list - "person id":[list of indexes]
        self.person_label_dict = dict()
        self.ct_file_list = []
        self.label_list = []

        for k in range(2):
            dataset_path_k = os.path.join(dataset_path, str(k))
            for index_file, file_name in enumerate(os.listdir(dataset_path_k)):
                if terminate: 
                    if index_file > terminate:
                        break
                file_path = os.path.join(dataset_path_k, file_name)
                ct_person = np.load(file_path)

                for i in range(3, ct_person.shape[0] - 3):
                    ct = rescale_image(ct_person[i,:,:], output_size=self.resolution)
                    distance = ct.max() - ct.min()
                    ct_normalized = (ct - ct.min()) / distance
                    self.ct_file_list.append(ct_normalized)
                    self.label_list.append(k)

        #weigths for balanced sampling
        self.label_list = torch.LongTensor(self.label_list)
        class0_number = ( self.label_list == 0).sum()
        class1_number = (self.label_list == 1).sum()
        p0 = class0_number / (class0_number + class1_number)
        p1 = class1_number / (class0_number + class1_number)
        self.weights = torch.zeros_like(self.label_list)
        self.weights = self.weights.type(torch.FloatTensor)
        index0 = self.label_list == 0
        self.weights[index0] = p1.item()
        index1 = self.label_list == 1
        self.weights[index1] = p0.item()

    def __len__(self):
            return len(self.ct_file_list)

    def __getitem__(self, index):
            return self.ct_file_list[index], self.label_list[index]


class Regressor(nn.Module):
    def __init__(self, d_model, latent=32):
        super().__init__() 
        self.fc1 = nn.Linear(d_model, latent)
        #self.bn1 = nn.BatchNorm1d(latent)
        self.bn1 = nn.LayerNorm(latent)
        self.fc2 = nn.Linear(latent, latent)
        #self.bn2 = nn.BatchNorm1d(latent)
        self.bn2 = nn.LayerNorm(latent)
        self.fc3 = nn.Linear(latent, latent)
        #self.bn3 = nn.BatchNorm1d(latent)
        self.bn3 = nn.LayerNorm(latent)
        self.fc4 = nn.Linear(latent, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)
        return x


def train_val_loaders(dataset, batch_size=20, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    weights = copy(dataset.weights)
    weights[val_idx] = 0
    number_of_batches = 2 * weights.shape[0] // batch_size
    sampler = WeightedRandomSampler(weights, 300*batch_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    val_sampler = SubsetRandomSampler(indices=val_idx)
    val_loader = DataLoader(dataset, batch_size=batch_size)
    return train_loader, val_loader


def validation_classification(data_loader, model, device="cpu"):
    regressor.eval()
    model.eval()
    predict_list = []
    label_list = []
    predict_original_list = []
    
    for image, label in train_loader:
        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        label = label.to(device)
        image = image.unsqueeze(1)
        image = image.to(device)
        with torch.no_grad():
            latent = model.encoder.encode(image, lod=lod)
        latent = latent.squeeze(1)
        with torch.no_grad():
            predict = regressor(latent)
        predict = predict.squeeze(1)
        predict_original = copy(predict.cpu())
        predict_original_list.append(predict_original)
        predict[predict < 0] = 0
        predict[predict > 1] = 1
        predict = torch.round(predict).cpu()
        predict_list.append(predict)
        label = label.type(torch.LongTensor).cpu()
        label_list.append(label)
    label_list = torch.hstack(label_list)
    predict_list = torch.hstack(predict_list)
    predict_original_list = torch.hstack(predict_original_list)
    result_dict = classification_report(label_list, predict_list, output_dict=True, zero_division=0)
    #predict_original_list[predict_original_list < 0] = 0
    #predict_original_list[predict_original_list > 1] = 1
    roc_auc = roc_auc_score(label_list, predict_original_list)
    return result_dict["accuracy"], roc_auc


if __name__ == "__main__":
    device = "cuda:1"
    batch_size = 20
    freeze_encoder = True
    resolution_power = 7
    resolution = resolution_power**2
    dataset_path_train = "/ayb/vol3/datasets/COVID19_1110_multi"
    dataset_path_val = "/ayb/vol1/kruzhilov/datasets/labeled_lungs/val"
    weights_path = 'weights/weights_ct256/model128_6layers_steps30plus.pth'

    covid_dataset = LungsCOVID(dataset_path_train, terminate=1500, resolution=128) 
 
    train_loader, val_loader = train_val_loaders(covid_dataset, batch_size=batch_size, val_split=0.2)

    # sum_1 = 0
    # sum_0 = 0
    # for _, label in val_loader:
    #     sum_1 += label.sum().item()
    #     sum_0 += (1-label).sum().item()
    # rate = sum_1 / sum_0
    
    print('')
    print("batch size:", batch_size)
    print("freeze encoder:", freeze_encoder)
    print('dataset size:', len(covid_dataset))#
    
    model = Model(channels=1, device=device, layer_count=6, latent_size=128)
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    regressor = Regressor(d_model=128)
    regressor.to(device)            
    lod = resolution_power - 2 

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': regressor.parameters()}], lr=0.0001, weight_decay=0.000)
    loss_f = nn.MSELoss()

    optimum_val_err = None
    for epoch in range(200):
        loss_list = []
        regressor.train()
        model.train()

        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.type(torch.FloatTensor)
            label = label.type(torch.FloatTensor)
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
        accuracy, mean_roc_auc = validation_classification(val_loader, model, device=device)
        print(epoch, train_err, accuracy, mean_roc_auc)

    # x, label = covid_dataset.__getitem__(0)
    # print(x.shape)
    # print(x.min(), x.max())

    # fig = plt.figure()
    # plt.axis('off') 
    # plt.imshow(x, cmap=plt.cm.gray)
    # plt.show()
# %%
