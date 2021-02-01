import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

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
            self.image_list.append(image_tensor)
    
    def __getitem__(self, idx):
        return self.image_list[idx]

    def __len__(self):
        return len(self.image_list)


def train_epoch(model, lungs_data_loader, encoder_optimizer, decoder_optimizer, current_lod, blend_factor=1):
    loss_d_list, loss_g_list, loss_lae_list = [], [], []
    for image_tensor in lungs_data_loader:
        if blend_factor < 1:
            needed_resolution = image_tensor.shape[1]
            x_prev = F.avg_pool2d(image_tensor, 2, 2)
            x_prev_2x = F.interpolate(x_prev, needed_resolution)
            image_tensor = image_tensor * blend_factor + x_prev_2x * (1.0 - blend_factor)
        image_tensor.requires_grad = True
        image_tensor = image_tensor.to(device)

        encoder_optimizer.zero_grad()
        loss_d = model(x=image_tensor, lod=current_lod, blend_factor=blend_factor, d_train=True, ae=False)
        loss_d_list.append(loss_d.item())
        loss_d.backward()
        encoder_optimizer.step()

        decoder_optimizer.zero_grad()   
        loss_g = model(x=image_tensor, lod=current_lod, blend_factor=blend_factor, d_train=False, ae=False)
        loss_g_list.append(loss_g.item())
        loss_g.backward()
        decoder_optimizer.step()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss_lae = model(x=image_tensor, lod=current_lod, blend_factor=blend_factor, d_train=False, ae=True)
        loss_lae_list.append(loss_lae.item())
        loss_lae.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        
    mean_loss_d = sum(loss_d_list) / len(loss_d_list)
    mean_loss_g = sum(loss_g_list) / len(loss_g_list)
    mean_loss_lae = sum(loss_lae_list) / len(loss_lae_list)
    return mean_loss_d, mean_loss_g, mean_loss_lae


if __name__ == "__main__":
    dataset_path = "/ayb/vol1/kruzhilov/lungs_images/"
    resolution_power = 6
    lung_dataset = CTDataset(dataset_path, resolution=2**resolution_power)
    print('dataset size:', len(lung_dataset))
    #exit()
    batch_size = 300
    lungs_data_loader = DataLoader(dataset=lung_dataset, shuffle=False, batch_size=batch_size, num_workers=2)
    device = "cuda:1"

    model = Model(channels=1, device=device, layer_count=5)
    model = model.to(device)
    model.load_state_dict(torch.load('weights/model32_5layers.pth'))

    decoder = model.decoder
    encoder = model.encoder
    mapping_d = model.mapping_d
    mapping_f = model.mapping_f

    decoder_optimizer = torch.optim.Adam([
        {'params': decoder.parameters()},
        {'params': mapping_f.parameters()}
    ], lr=0.001, weight_decay=0)

    encoder_optimizer = torch.optim.Adam([
        {'params': encoder.parameters()},
        {'params': mapping_d.parameters()},
    ], lr=0.001, weight_decay=0)


    current_lod = resolution_power - 2
    save_model_path = 'weights/model__5layers.pth'

    for epoch in range(300):
        current_blend_factor = min(1, 0.5 + epoch*0.1)
        if epoch > 0:
            old_d, old_g, old_lae = mean_loss_d, mean_loss_g, mean_loss_lae
        mean_loss_d, mean_loss_g, mean_loss_lae = \
            train_epoch(model, lungs_data_loader, encoder_optimizer, decoder_optimizer, \
                 current_lod=current_lod, blend_factor=current_blend_factor)       
        print(epoch, mean_loss_d, mean_loss_g, mean_loss_lae)
        # if epoch > 0 and mean_loss_d > 5 * old_d:
        #     model.load_state_dict(torch.load(save_model_path))
        #     continue
        torch.save(model.state_dict(), save_model_path)    
    # image = model.generate(lod=2, blend_factor=1, device=model.device)
    # image = image[0, 0, :, :].detach().cpu().numpy()
    # plt.imshow(image, cmap=plt.cm.gray)
    # plt.show()

# %%
