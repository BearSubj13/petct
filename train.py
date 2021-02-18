import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from losses import reconstruction

from model import Model


class CTDataset(Dataset):
    def __init__(self, dataset_path, resolution=None):
        super().__init__()
        self.image_list = []
        for file_name in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, file_name)
            image_tensor = torch.load(file_path)
            image_tensor = image_tensor.unsqueeze(0)#.unsqueeze(0)
            if resolution:
               image_tensor = F.interpolate(image_tensor, size=resolution)
               image_tensor = image_tensor.permute(0, 2, 1)
               image_tensor = F.interpolate(image_tensor, size=resolution ) 
               image_tensor = image_tensor.permute(0, 2, 1)
            #normalize to [-0.5, 0.5]   
            #image_tensor = image_tensor - 0.5        
            self.image_list.append(image_tensor)
    
    def __getitem__(self, idx):
        return self.image_list[idx]

    def __len__(self):
        return len(self.image_list)


def augmentation(image_tensor, p_augment=0.5):
    if np.random.uniform() < p_augment:
        #dims = [[2], [3], [2, 3]]
        #dim = np.random.choice(dims)
        dim = [3]
        if np.random.uniform() < 0.5:
            image_tensor = torch.flip(image_tensor, dims=dim) 
        #random_coeff = np.random.uniform(low=0.9)
        #if np.random.uniform() < 0.8:
        #    image_tensor = random_coeff*image_tensor 
        return image_tensor
    else:
        return image_tensor


def train_epoch(model, lungs_data_loader, encoder_optimizer, discriminator_optimizer, decoder_optimizer,
 current_lod, blend_factor=1, freeze_previous_layers=None, lambda_reconstruct=0,
  p_augment=0.5, r1_gamma=100, boundary=0.0):
    model.train()
    loss_d_list, loss_g_list, loss_lae_list = [], [], []
    loss_d_reconst_list, loss_image_reconstruction_list = [], []
    loss_g_reconst_list, loss_latent_reconstruction_list = [], []
    loss_boundary_list = []
    for image_tensor in lungs_data_loader:
        image_tensor = augmentation(image_tensor, p_augment=p_augment)

        if blend_factor < 1:
            needed_resolution = image_tensor.shape[1]
            x_prev = F.avg_pool2d(image_tensor, 2, 2)
            x_prev_2x = F.interpolate(x_prev, needed_resolution)
            image_tensor = image_tensor * blend_factor + x_prev_2x * (1.0 - blend_factor)
        image_tensor.requires_grad = True
        image_tensor = image_tensor.to(device)

        #discriminator for latent = F(z), z~N(0,1)
        encoder_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()
        loss_d = model(x=image_tensor, lod=current_lod, blend_factor=blend_factor, \
                      d_train=True, ae=False, freeze_previous_layers=freeze_previous_layers,\
                      r1_gamma=r1_gamma)
        loss_d_list.append(loss_d.item())
        loss_d.backward()
        encoder_optimizer.step()
        discriminator_optimizer.step()

        #discriminator for latent = Encoder(Generator(Encoder(image_real)))
        #encoder_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()
        loss_d_reconstr = model.autoencoder_discriminator(x=image_tensor, lod=current_lod, \
                                                blend_factor=blend_factor, r1_gamma=r1_gamma)
        loss_d_reconst_list.append(loss_d_reconstr.item())
        loss_d_reconstr.backward()
        #encoder_optimizer.step()
        discriminator_optimizer.step()

        #generator
        decoder_optimizer.zero_grad()   
        loss_g = model(x=image_tensor, lod=current_lod, blend_factor=blend_factor, \
                     d_train=False, ae=False, freeze_previous_layers=freeze_previous_layers, \
                     r1_gamma=r1_gamma)
        loss_g_list.append(loss_g.item())
        loss_g.backward()
        decoder_optimizer.step()

        #lae - latent autoencoder [latent - Encoder(Generator(latent))]ˆ2
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()
        loss_lae = model(x=image_tensor, lod=current_lod, blend_factor=blend_factor, \
                         d_train=True, ae=True, freeze_previous_layers=freeze_previous_layers,
                         r1_gamma=r1_gamma)
        loss_lae_list.append(loss_lae.item())
        loss_lae.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        discriminator_optimizer.step()

        # decoder_optimizer.zero_grad()
        # loss_boundary = model.border_penalty(x=image_tensor, lod=current_lod, blend_factor=blend_factor, \
        #                     freeze_previous_layers=freeze_previous_layers)
        # loss_boundary_list.append(loss_boundary.item())
        # loss_boundary = boundary * loss_boundary
        # loss_boundary.backward()
        # decoder_optimizer.step()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss_image_reconstruction, loss_latent_reconstruction, loss_g_reconstr = \
            model.reciprocity(x=image_tensor, lod=current_lod, blend_factor=blend_factor, loss=reconstruction, freeze_previous_layers=freeze_previous_layers)
        loss_latent_reconstruction_list.append(loss_latent_reconstruction.item())
        loss_image_reconstruction_list.append(loss_image_reconstruction.item())
        loss_g_reconst_list.append(loss_g_reconstr.item())
        reconstr_loss = lambda_reconstruct * loss_image_reconstruction
        reconstr_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        
    mean_loss_d = sum(loss_d_list) / len(loss_d_list)
    mean_loss_g = sum(loss_g_list) / len(loss_g_list)
    mean_loss_lae = sum(loss_lae_list) / len(loss_lae_list)
    mean_loss_image_reconstruction = sum(loss_image_reconstruction_list) / len(loss_image_reconstruction_list)
    mean_loss_latent_reconstruction = sum(loss_latent_reconstruction_list) / len(loss_latent_reconstruction_list)
    mean_loss_d_reconstr = sum(loss_d_reconst_list) / len(loss_d_reconst_list)
    mean_loss_g_reconstr = sum(loss_g_reconst_list) / len(loss_g_reconst_list)
    #mean_loss_boundary = sum(loss_boundary_list) / len(loss_boundary_list)
    result = {'d':mean_loss_d, 'g':mean_loss_g, 'lae':mean_loss_lae, \
              'd_rec':mean_loss_d_reconstr, 'g_rec':mean_loss_g_reconstr, \
              'lae_rec':mean_loss_latent_reconstruction, \
              'mse_rec':mean_loss_image_reconstruction}#, 'boundary':mean_loss_boundary}
    return result


def validation_epoch(model, lungs_data_loader, current_lod):
    model.eval()
    for image_tensor in lungs_data_loader:
        image_tensor = image_tensor.to(device)
        loss_image_reconstruction, loss_latent_reconstruction, loss_g_reconstr = \
            model.reciprocity(x=image_tensor, lod=current_lod, blend_factor=1, loss=reconstruction)
        mse = loss_image_reconstruction.item()    
        if mse > 0:
            psnr = -10*np.log10(mse)    
        else:
            psnr = 100
    return psnr


if __name__ == "__main__":
    dataset_path = "/ayb/vol1/kruzhilov/lungs_images/"
    dataset_path_val = "/ayb/vol1/kruzhilov/lungs_images_val/"
    resolution_power = 2
    load_model_path = None#'weights/model4_5layers.pth'
    save_model_path = 'weights/model4_5layers.pth'
    r1_gamma = 50
    mse_penalty = 0.1
    weigt_decay = 0.0001
    p_augment = 0.8
    boundary = 0.0
    batch_size = 200
    blending_step = 0.05 
    lr = 0.001 

    device = "cuda:1"
    print('resolution:', 2**resolution_power)
    print('loaded from:', load_model_path)
    print("batch size:", batch_size)

    lung_dataset = CTDataset(dataset_path, resolution=2**resolution_power)
    lung_dataset_val = CTDataset(dataset_path_val, resolution=2**resolution_power)
    print('train dataset size:', len(lung_dataset), 'validation:', len(lung_dataset_val))
    #exit()
    lungs_data_loader = DataLoader(dataset=lung_dataset, shuffle=True, batch_size=batch_size, num_workers=2)
    lungs_data_loader = DataLoader(dataset=lung_dataset_val, shuffle=False, batch_size=batch_size)

    model = Model(channels=1, device=device, layer_count=5)
    model = model.to(device)
    if load_model_path:
        model.load_state_dict(torch.load(load_model_path, map_location=device))
    model.train()

    decoder = model.decoder
    encoder = model.encoder
    mapping_d = model.mapping_d
    mapping_f = model.mapping_f

    decoder_optimizer = torch.optim.Adam([
        {'params': decoder.parameters()},
        {'params': mapping_f.parameters()}
    ], lr=2*lr, weight_decay=0)

    encoder_optimizer = torch.optim.Adam([
        {'params': encoder.parameters()},
    ], lr=2*lr, weight_decay=weigt_decay)

    discriminator_optimizer = torch.optim.Adam([
        {'params': mapping_d.parameters()},
    ], lr=lr, weight_decay=0)

    current_lod = resolution_power - 2

    for epoch in range(400):
        current_blend_factor = 1#min(1, 0.01 + epoch*blending_step)
        freeze_previous_layers = None
        loss = train_epoch(model, lungs_data_loader, encoder_optimizer, discriminator_optimizer, decoder_optimizer, \
                 current_lod=current_lod, blend_factor=current_blend_factor, \
                 freeze_previous_layers=freeze_previous_layers,
                 lambda_reconstruct=mse_penalty, p_augment=p_augment, boundary=boundary)       

        psnr = validation_epoch(model, lungs_data_loader, current_lod)

        format_output = "{0:3d}  d:{1:2.5f}, g:{2:2.5f}, lae:{3:2.5f}    lae rec:{4:2.5f}, mse:{5:2.5f}, psnr:{6:3.1f}, d rec:{7:2.5f}". \
            format(epoch, loss['d'], loss['g'], loss['lae'], loss['lae_rec'], \
                 loss['mse_rec'], psnr, loss['d_rec'])#, loss['boundary'])         
        print(format_output)

        if loss['mse_rec'] < 0.1 and epoch > 10:
            torch.save(model.state_dict(), save_model_path)
        

# %%
