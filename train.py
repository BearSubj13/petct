#%%
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt

from model import Model


class CTDataset(Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.image_list = []
        for file_name in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, file_name)
            image_tensor = torch.load(file_path)
            image_tensor = image_tensor.unsqueeze(0)#.unsqueeze(0)
            self.image_list.append(image_tensor)
    
    def __getitem__(self, idx):
        return self.image_list[idx]

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    dataset_path = "/ayb/vol1/kruzhilov/lungs_images/"
    lung_dataset = CTDataset(dataset_path)
    batch_size = 16
    lungs_data_loader = DataLoader(dataset=lung_dataset, shuffle=False, batch_size=batch_size, num_workers=2)
    device = "cuda:1"

    model = Model(channels=1, device=device)
    model = model.to(device)
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


    for epoch in range(20):
        loss_d_list, loss_g_list, loss_lae_list = [], [], []
        for image_tensor in lungs_data_loader:
            image_tensor.requires_grad = True
            image_tensor = image_tensor.to(device)

            encoder_optimizer.zero_grad()
            loss_d = model(x=image_tensor, lod=2, blend_factor=1, d_train=True, ae=False)
            loss_d_list.append(loss_d.item())
            loss_d.backward()
            encoder_optimizer.step()

            decoder_optimizer.zero_grad()   
            loss_g = model(x=image_tensor, lod=2, blend_factor=1, d_train=False, ae=False)
            loss_g_list.append(loss_g.item())
            loss_g.backward()
            decoder_optimizer.step()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss_lae = model(x=image_tensor, lod=2, blend_factor=1, d_train=False, ae=True)
            loss_lae_list.append(loss_lae.item())
            loss_lae.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
           
        print(sum(loss_d_list) / len(loss_d_list), sum(loss_g_list) / len(loss_g_list), sum(loss_lae_list) / len(loss_lae_list))

    torch.save(model.state_dict(), 'model.pth')    
    image = model.generate(lod=2, blend_factor=1, device=model.device)
    image = image[0, 0, :, :].detach().cpu().numpy()
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()

# %%
