import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import piq

from losses import reconstruction, reconstruction_l1, lasso
from model import Model


class CTDataset(Dataset):
    def __init__(self, dataset_path, resolution=None):
        super().__init__()
        self.image_list = []
        for file_name in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, file_name)
            image_tensor = torch.load(file_path)
            image_tensor = image_tensor.unsqueeze(0)#.unsqueeze(0)
            if torch.isnan(image_tensor).any() or torch.isinf(image_tensor).any():
                print(file_name)
                continue
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
        #loss_d_reconstr.backward()
        #encoder_optimizer.step()
        #discriminator_optimizer.step()

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

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss_image_reconstruction, loss_latent_reconstruction, loss_g_reconstr, _ = \
            model.reciprocity(x=image_tensor, lod=current_lod, blend_factor=blend_factor, \
                 loss=reconstruction, freeze_previous_layers=freeze_previous_layers)
        loss_latent_reconstruction_list.append(loss_latent_reconstruction.item())
        loss_image_reconstruction_list.append(loss_image_reconstruction.item())
        loss_g_reconst_list.append(loss_g_reconstr.item())
        reconstr_loss = lambda_reconstruct * loss_image_reconstruction
        reconstr_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        #print(loss_d.item(), loss_d_reconstr.item(), loss_g.item(), loss_lae.item(), loss_image_reconstruction.item())
        #if torch.isnan(loss_d):
        #    print(image_tensor.sum())
        #    exit()
        
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


def image_normalization(image):
    image[image < -0.05] = -0.05
    image[image > 1.05] = 1.05
    min_pixel_value = image.min(dim=3)[0].min(dim=2)[0]
    max_pixel_value = image.max(dim=3)[0].max(dim=2)[0]  
    diff = max_pixel_value - min_pixel_value

    min_pixel_value = min_pixel_value.unsqueeze(2).unsqueeze(3)
    min_pixel_value = min_pixel_value.repeat(1, 1, image.shape[2], image.shape[3])
    diff = diff.unsqueeze(2).unsqueeze(3)
    diff = diff.repeat(1, 1, image.shape[2], image.shape[3])
    non_zero_diff = diff != 0
    
    image[non_zero_diff] = (image[non_zero_diff] - min_pixel_value[non_zero_diff]) / diff[non_zero_diff]
    image[diff == 0] = 0
    return image


def ssim_metric(image_original, image_reconstructed):
    size = image_original.shape[-1]
    if size < 8:
        return None
    elif size == 8 or size == 16:
        kernel_size = 3
    elif size == 32:
        kernel_size = 5
    elif size == 64:
        kernel_size = 7
    else:
        kernel_size = 9

    ssim_loss = piq.SSIMLoss(kernel_size=kernel_size)

    image_reconstructed = image_normalization(image_reconstructed)
    ssim = ssim_loss(image_original, image_reconstructed)
    return ssim 


def psnr_metric(image_original, image_reconstructed):
    image_reconstructed = image_reconstructed.squeeze(1) 
    image_original = image_original.squeeze(1)  
    mse = torch.mean((image_reconstructed - image_original)**2, dim=[-1, -2])
    psnr = 100*torch.ones_like(mse).to(image_original.device)
    psnr[mse > 0] = -10*torch.log10(mse[mse > 0]) 
    return psnr.mean()

def validation_epoch(model, lungs_data_loader, current_lod):
    model.eval()
    ssim_list = []
    psnr_list = []

    with torch.no_grad():
        for image_tensor in lungs_data_loader:
            image_tensor = image_tensor.to(device)
            loss_image_reconstruction, loss_latent_reconstruction,\
                loss_g_reconstr, image_reconstructed = \
                model.reciprocity(x=image_tensor, lod=current_lod, blend_factor=1, loss=reconstruction)
            psnr = psnr_metric(image_tensor, image_reconstructed)
            psnr_list.append(psnr)

            ssim = ssim_metric(image_tensor, image_reconstructed)
            ssim_list.append(ssim)

        psnr = sum(psnr_list) / len(psnr_list)
        ssim = sum(ssim_list) / len(ssim_list)
    return psnr, ssim


if __name__ == "__main__":
    dataset_path = "/ayb/vol1/kruzhilov/lungs_images/"
    dataset_path_val = "/ayb/vol1/kruzhilov/lungs_images_val/"
    resolution_power = 6
    load_model_path = 'weights/model64_5layers_2.pth'
    save_model_path = 'weights/model64_5layers.pth'
    r1_gamma = 70
    mse_penalty = 0.1
    weight_decay = 0.001
    boundary = 0.0
    batch_size = 100
    blending_step = None#0.04    
    lr = 0.0001

    device = "cuda:2"
    print('resolution:', 2**resolution_power)
    print('loaded from:', load_model_path)
    print("batch size:", batch_size)
    print('lr:', lr)
    print('r1 gamma:', r1_gamma)
    print('encoder weight decay:', weight_decay)
    print('mse penalty:', mse_penalty)

    lung_dataset = CTDataset(dataset_path, resolution=2**resolution_power)
    lung_dataset_val = CTDataset(dataset_path_val, resolution=2**resolution_power)
    print('train dataset size:', len(lung_dataset), 'validation:', len(lung_dataset_val))
    #exit()
    lungs_data_loader = DataLoader(dataset=lung_dataset, shuffle=True, batch_size=batch_size)
    lungs_data_loader_val = DataLoader(dataset=lung_dataset_val, shuffle=False, batch_size=batch_size)

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
    ], lr=2*lr, weight_decay=weight_decay)

    discriminator_optimizer = torch.optim.Adam([
        {'params': mapping_d.parameters()},
    ], lr=lr, weight_decay=0)

    current_lod = resolution_power - 2

    for epoch in range(500):
        if blending_step:
            current_blend_factor = min(1, 0.01 + epoch*blending_step)
        else:
            current_blend_factor = 1
        freeze_previous_layers = None
        loss = train_epoch(model, lungs_data_loader, encoder_optimizer, discriminator_optimizer, decoder_optimizer, \
                 current_lod=current_lod, blend_factor=current_blend_factor, \
                 freeze_previous_layers=freeze_previous_layers,
                 lambda_reconstruct=mse_penalty, boundary=boundary)       

        psnr, ssim = validation_epoch(model, lungs_data_loader, current_lod)

        format_output = "{0:3d}  d:{1:2.5f}, g:{2:2.5f}, lae:{3:2.5f}    lae rec:{4:2.5f}, mse:{5:2.5f},   psnr:{6:3.1f}, ssim:{7:1.5f}". \
            format(epoch, loss['d'], loss['g'], loss['lae'], loss['lae_rec'], \
                 loss['mse_rec'], psnr, ssim)#, loss['boundary'])         
        print(format_output)

        if loss['mse_rec'] < 0.03 and epoch > 10:
            torch.save(model.state_dict(), save_model_path)
        

# %%
