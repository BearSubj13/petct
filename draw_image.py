#%%
import torch
import matplotlib.pyplot as plt

from model import Model
from train import CTDataset

if __name__ == "__main__":
    resolution_power = 2
    dataset_path = "/ayb/vol1/kruzhilov/lungs_images/"
    lung_dataset = CTDataset(dataset_path, resolution=2**resolution_power)
    device = "cpu"# "cuda:2"
    model = Model(channels=1, device=device, layer_count=5)
    model = model.to(device)
    model.load_state_dict(torch.load('weights/model__5layers.pth', map_location=device)) #strict=False
    model.eval()

    generate_mode = False
    if generate_mode:
        z =  torch.randn(2*(resolution_power-1), 128)
        image = model.generate(lod=resolution_power - 2, blend_factor=1.0, z=z, device=model.device, noise=True)
        image = image[0, 0, :, :].detach().cpu().numpy()
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()
    else:
        item = 0
        image = lung_dataset.__getitem__(item).squeeze(0)
        print('original image')
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()
        image = lung_dataset.__getitem__(item).unsqueeze(0)
        image_gen = model.autoencoder(x=image, lod=resolution_power - 2, device=device)
        image = image_gen[0,:,:,:].squeeze().detach().numpy()
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()


# %%
