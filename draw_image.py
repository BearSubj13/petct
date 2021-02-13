#%%
import torch
import matplotlib.pyplot as plt
import numpy as np

from model import Model
from train import CTDataset, augmentation


def image_normalization(image):
    image[image < -0.05] = -0.05
    image[image > 1.05] = 1.05
    min_pixel_value = image.min() 
    max_pixel_value = image.max() 
    diff = max_pixel_value - min_pixel_value
    image = (image - min_pixel_value) / diff
    return image

if __name__ == "__main__":
    resolution_power = 2
    dataset_path = "/ayb/vol1/kruzhilov/lungs_images_val"
    #lung_dataset = CTDataset(dataset_path, resolution=2**resolution_power)
    #print(len(lung_dataset))
    device = "cpu"# "cuda:2"
    generate_mode = True
    model_path = 'weights/model4_5layers.pth'

    model = Model(channels=1, device=device, layer_count=5)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)) #strict=False
    model.eval()

    if generate_mode:
        z =  torch.randn(2*(resolution_power-1), 128)
        image = model.generate(lod=resolution_power - 2, blend_factor=1.0, z=z, device=model.device, noise=True)
        image = image[0, 0, :, :].detach().cpu().numpy()
        print(image.min().item(), image.max().item())
        image = image_normalization(image)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()
    else:
        lung_dataset = CTDataset(dataset_path, resolution=2**resolution_power)
        item = np.random.randint(len(lung_dataset))
        image = lung_dataset.__getitem__(item)
        image = image.unsqueeze(0)
        image = augmentation(image, p_augment=1)
        image_gen, _ = model.autoencoder(x=image, lod=resolution_power - 2, device=device)
        print(image_gen.min().item(), image_gen.max().item())
        image_gen = image_gen[0,:,:,:].squeeze().detach().numpy()
        image_gen = image_normalization(image_gen)

        image = image.squeeze().detach().numpy()
        print('original image')
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()
        plt.imshow(image_gen, cmap=plt.cm.gray)
        plt.show()
        error = np.abs(image - image_gen).mean()
        print("error:", error)


    # %%
