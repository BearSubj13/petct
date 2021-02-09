#%%
import torch
import matplotlib.pyplot as plt
import numpy as np

from model import Model
from train import CTDataset, augmentation

if __name__ == "__main__":
    resolution_power = 3
    dataset_path = "/ayb/vol1/kruzhilov/lungs_images/"
    lung_dataset = CTDataset(dataset_path, resolution=2**resolution_power)
    device = "cpu"# "cuda:2"
    model = Model(channels=1, device=device, layer_count=5)
    model = model.to(device)
    model.load_state_dict(torch.load('weights/model8_5layers.pth', map_location=device)) #strict=False
    model.eval()

    generate_mode = True
    if generate_mode:
        z =  torch.randn(2*(resolution_power-1), 128)
        image = model.generate(lod=resolution_power - 2, blend_factor=1.0, z=z, device=model.device, noise=True)
        image = image[0, 0, :, :].detach().cpu().numpy()
        image_gen[image_gen < 0] = 0
        image_gen[image_gen > 1] = 1
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()
    else:
        item = 0
        image = lung_dataset.__getitem__(item)
        image = image.unsqueeze(0)
        #image = augmentation(image, p_augment=1)
        image_gen = model.autoencoder(x=image, lod=resolution_power - 2, device=device)
        image_gen = image_gen[0,:,:,:].squeeze().detach().numpy()
        image_gen[image_gen < 0] = 0
        image_gen[image_gen > 1] = 1

        image = image.squeeze().detach().numpy()
        print('original image')
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()
        plt.imshow(image_gen, cmap=plt.cm.gray)
        plt.show()
        error = np.abs(image - image_gen).mean()
        print("error:", error)


# %%
