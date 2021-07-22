#%%

import os
#from PIL.Image import blend
import numpy as np
import matplotlib.pyplot as plt
import bisect

import torch
from torch import nn
from torch.utils.data import DataLoader
#from albumentations import VerticalFlip #as alb

from dataset_with_labels import LungsLabeled
from model import Model
from luna16 import train_val_split_luna
from utils.decoder import Decoder1
from unet.unet_model import UNet
from luna16 import coordinates_metrics, find_stars


def train_epoch(model, decoder, data_loader, encoder_optimizer, decoder_optimizer, device, lod, freeze_encoder=False):
    model.train()
    decoder.train()
    pos_weigth = torch.FloatTensor([2.0])
    pos_weigth = pos_weigth.to(device)
    loss_f = nn.BCEWithLogitsLoss(pos_weight=pos_weigth)
    loss_list = []
    for image, heat_map, _ in data_loader:
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


def train_epoch_unet(unet, data_loader, optimizer, loss_f):
    device = next(unet.parameters()).device
    unet.train()
    loss_list = []
    for image, heat_map, _ in data_loader:
        optimizer.zero_grad()
        image = image.type(torch.FloatTensor)
        image = image.unsqueeze(1)
        image = image.to(device)
        heat_map = heat_map.to(device)
        image_reconstructed = unet(image).squeeze(1)
        image_reconstructed = image_reconstructed.squeeze(1)
        assert image_reconstructed.shape == heat_map.shape

        loss = loss_f(image_reconstructed, heat_map)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
    mean_loss = sum(loss_list) / len(loss_list)
    return mean_loss


def eval_epoch_unet(unet, data_loader, loss_f):
    device = next(unet.parameters()).device
    unet.train()
    loss_list = []
    for image, heat_map, _ in data_loader:
        image = image.type(torch.FloatTensor)
        image = image.unsqueeze(1)
        image = image.to(device)
        heat_map = heat_map.to(device)
        with torch.no_grad():
            image_reconstructed = unet(image).squeeze(1)
        image_reconstructed = image_reconstructed.squeeze(1)
        assert image_reconstructed.shape == heat_map.shape

        loss = loss_f(image_reconstructed, heat_map)
        loss_list.append(loss.item())
    mean_loss = sum(loss_list) / len(loss_list)
    return mean_loss


def eval_epoch(model, decoder, data_loader, device, lod):
    model.eval()
    decoder.eval()
    loss_f = nn.BCEWithLogitsLoss()
    loss_list = []
    for image, heat_map, _ in data_loader:
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


def train_procedure():
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


def train_procedure_unet():
    resolution_power = 8
    load_model_path = 'weights/luna_heatmap/unet256.pth'
    save_model_path = 'weights/luna_heatmap/unet256_pos1_8.pth'
    weight_decay = 0.0001
    batch_size = 32
    lr = 0.00005
    device = "cuda:3"
    dataset_path = "/ayb/vol1/kruzhilov/datasets/luna16_heatmap/resolution256_largeheatmap/"
    pos_weight = torch.FloatTensor([1.9])
    pos_weight = pos_weight.to(device)
    loss_f = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    unet = UNet(n_channels=1, n_classes=1)
    unet = unet.to(device)
    if load_model_path:
        unet.load_state_dict(torch.load(load_model_path, map_location=device))

    train_dataset, val_dataset = train_val_split_luna(dataset_path, val_rate=0.1)
    print("train len:", len(train_dataset), ", val len:", len(val_dataset))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam([{'params': unet.parameters()}], lr=lr, weight_decay=0)

    for epoch in range(200):    
        loss_train = train_epoch_unet(unet, train_data_loader, optimizer, loss_f)
        loss_val = eval_epoch_unet(unet, train_data_loader, loss_f)
        print(epoch, loss_train, loss_val)
        if (epoch >= 10) and ((epoch % 10) == 0):
            if save_model_path:
                torch.save(unet.state_dict(), save_model_path)
                test_precision_recall(device, dataset_path, save_model_path)
    
    if save_model_path:
        torch.save(unet.state_dict(), save_model_path)


def test_precision_recall(device, dataset_path, load_model_path):
    batch_size = 1

    train_dataset, val_dataset = train_val_split_luna(dataset_path, val_rate=0.1)
    print("train len:", len(train_dataset), ", val len:", len(val_dataset))
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
   
    unet = UNet(n_channels=1, n_classes=1)
    unet = unet.to(device)
    unet.load_state_dict(torch.load(load_model_path, map_location=device))

    prec_list = []
    recall_list = []
    err_list = []

    for image, _, coordinates in val_data_loader:
        coordinates = coordinates.reshape([-1, 3])
        image = image.type(torch.FloatTensor)
        image = image.unsqueeze(1)
        image = image.to(device)
        with torch.no_grad():
            image_reconstructed = unet(image).squeeze(1)
        image_reconstructed = image_reconstructed.squeeze(0)
        image_reconstructed = image_reconstructed.cpu().numpy()

        xyr = find_stars(image_reconstructed)
        prec, recall, std_coord = coordinates_metrics(xyr, coordinates, efficient_radius=1.6)
        if std_coord is not None:
            err_list.append(std_coord)
        if std_coord is not None:
            prec_list.append(prec)
        recall_list.append(recall)
    
    mean_prec = sum(prec_list) / len(prec_list)
    mean_recall = sum(recall_list) / len(recall_list)
    mean_err = sum(err_list) / len(err_list)
    print("precision:", mean_prec, "recall:", mean_recall, "err:", mean_err, "pixel")


def extract_coordinates(coordinates_list, eps=1.0):
    triplet_list = []
    for k in range(len(coordinates_list) - 2):
        for i1 in range(coordinates_list[k].shape[0]):
            for i2 in range(coordinates_list[k + 1].shape[0]):
                for i3 in range(coordinates_list[k + 2].shape[0]):
                    xy1 = coordinates_list[k][i1, :2]
                    xy2 = coordinates_list[k + 1][i2, :2]
                    xy3 = coordinates_list[k + 2][i3, :2]
                    d12 = np.linalg.norm(xy1 - xy2)
                    d23 = np.linalg.norm(xy2 - xy3)
                    d13 = np.linalg.norm(xy1 - xy3)
                    if d12 < eps and d23 < eps and d13 < eps:
                        xy1 = coordinates_list[k][i1, :]
                        xy2 = coordinates_list[k + 1][i2, :]
                        xy3 = coordinates_list[k + 2][i3, :]
                        triplet_list.append([xy1, xy2, xy3])
   
    new_triplet_list = []
    fl = True
    for i, triplet1 in enumerate(triplet_list):
       for j in range(i+1, len(triplet_list)):
           triplet2 = triplet_list[j]
           if triplet1[-1][3] == triplet2[-2][3]:
               new_group = triplet1 + [triplet2[-1]]
               new_triplet_list.append(new_group)
               for j2 in range(i+2, len(triplet_list)):
                   new_triplet_list.append(triplet_list[j2])
               fl = False
               break
           else:
               new_triplet_list.append(triplet1)
       if not fl:
           break
    triplet_list = new_triplet_list

    return triplet_list


def look_for_tumors(device, dataset_path, load_model_path):
    batch_size = 1

    val_dataset = LungsLabeled(dataset_path, resolution=256, terminate=5, load_labels=True)
    print("dataset len:", len(val_dataset))
    #val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
   
    unet = UNet(n_channels=1, n_classes=1)
    unet = unet.to(device)
    unet.load_state_dict(torch.load(load_model_path, map_location=device))

    key = list(val_dataset.person_index_dict.keys())[1] #2
    print("id", key)

    coordinates_list = []
    ct_index_list = val_dataset.person_index_dict[key]
    z_list = []
    for ct_index in ct_index_list:
        z_list.append(val_dataset.label_list[ct_index]["slice"])
        ct_image = val_dataset[ct_index]
        ct_image = ct_image.unsqueeze(0)
        ct_image = ct_image.to(device)
        heatmap = unet(ct_image)
        heatmap = heatmap.squeeze(0).squeeze(0)
        heatmap = heatmap.cpu().detach().numpy()
        coordinates = find_stars(heatmap, threshold=0.1)
        if len(coordinates) > 0:
                new_column = np.zeros([coordinates.shape[0], 1])
                coordinates = np.hstack([coordinates, new_column])
                ct_image = ct_image.cpu().squeeze(0).squeeze(0).numpy()
                for i in range(coordinates.shape[0]):
                    x = coordinates[i, 0]
                    y = coordinates[i, 1]
                    r = coordinates[i, 2]
                    z = val_dataset.label_list[ct_index]["slice"]
                    coordinates[i, 3] = z
                    # print(x, y, z)
                    # if number_of_drawn_pictures < 20:
                    #     number_of_drawn_pictures = number_of_drawn_pictures + 1
                    #     fig, ax = plt.subplots()
                    #     plt.imshow(ct_image, cmap=plt.cm.gray)
                    #     circle = plt.Circle((x, y), 1.2*r, color='r', fill=False)
                    #     ax.add_artist(circle)
                    #     plt.show()
                #plt.imshow(heatmap, cmap=plt.cm.gray)
                #plt.show()
                #break
                coordinates_list.append(coordinates)
    
    triplet_list = extract_coordinates(coordinates_list, eps=3.0)
    #coordinates_chain = np.array(triplet_list[0])
    #average_coord = coordinates_chain.mean(axis=0)
    #print(average_coord)
    while True:
        new_triplet_list = []
        merged_triplet = False
        current_i = 0    
        for i in range(current_i, len(triplet_list)):
            for j in range(i+1, len(triplet_list)):
                if triplet_list[i][-1][3] == triplet_list[j][-2][3]:
                    new_triplet = triplet_list[i] + [triplet_list[j][-1]]
                    new_triplet_list.append(new_triplet)
                    for k in range(i + 2, len(triplet_list)):
                        new_triplet_list.append(triplet_list[k])
                    merged_triplet = True
                    current_i = i
                    break
            if merged_triplet:
                break
            else:
                new_triplet_list.append(triplet_list[i])

        if not merged_triplet:
            break
        triplet_list = new_triplet_list

    averaged_chain = []
    for coordinates_chain in triplet_list:
        numpy_chain = np.array(coordinates_chain)
        average_coord = numpy_chain.mean(axis=0)
        averaged_chain.append(average_coord)

    for coord in averaged_chain:
        k = bisect.bisect_left(z_list, coord[3])
        x = coord[0]
        y = coord[1]
        r = coord[2]
        ct_image = val_dataset[ct_index_list[k]].squeeze(0)
        _, ax = plt.subplots()
        print(x, y, coord[3])
        plt.imshow(ct_image, cmap=plt.cm.gray)
        circle = plt.Circle((x, y), 1.2*r, color='r', fill=False)
        ax.add_artist(circle)
        plt.show()
        

    return averaged_chain


if __name__ == "__main__":
    #the best results 
    #precision: 0.9132183908045979 recall: 0.7983425414364641
    #precision: 0.9531435349940689 recall: 0.7720994475138122
    #train_procedure_unet()
    #test_precision_recall(device="cuda:3", dataset_path = "/ayb/vol1/kruzhilov/datasets/luna16_heatmap/resolution256/", load_model_path = 'weights/luna_heatmap/unet256_pos1_8.pth')
    dataset_path="/ayb/vol1/kruzhilov/datasets/labeled_lungs_description/labeled_lungs_description_256/rostov"
    tumor_centers = look_for_tumors(device="cuda:3", dataset_path=dataset_path, load_model_path='weights/luna_heatmap/unet256_pos1_8.pth') #_pos1_8
    for tumor in tumor_centers:
         tumor[:3] = 2 * tumor[:3]
         print(tumor)

    # dataset_path = "/ayb/vol1/kruzhilov/datasets/labeled_lungs_description/labeled_lungs_description_256/train"
    # empty_list = []
    # for folder in os.listdir(dataset_path):
    #     list_of_items = os.listdir(os.path.join(dataset_path, folder))
    #     if "ct" not in list_of_items or "pi" not in list_of_items or "labeles" not in list_of_items:
    #         empty_list.append(folder)
    # print(empty_list)





# %%
