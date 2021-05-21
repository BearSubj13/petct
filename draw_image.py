#%%
import torch
import matplotlib.pyplot as plt
import numpy as np

from model import Model
from train import CTDataset, augmentation
from dataset_with_labels import LungsLabeled


def image_normalization(image):
    image[image < -0.05] = -0.05
    image[image > 1.05] = 1.05
    min_pixel_value = image.min() 
    max_pixel_value = image.max() 
    diff = max_pixel_value - min_pixel_value
    image = (image - min_pixel_value) / diff
    return image

if __name__ == "__main__":
    resolution_power = 6
    dataset_path = "/ayb/vol1/kruzhilov/datasets/labeled_lungs_description/train"
    device = "cpu"# "cuda:2"
    generate_mode = False
    model_path = 'weights/weights_pi/pi64_5layers.pth'

    model = Model(channels=2, device=device, layer_count=5)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)) #strict=False
    model.eval()

    if generate_mode:
        z = torch.randn(2*(resolution_power-1), 128)
        image = model.generate(lod=resolution_power - 2, blend_factor=1.0, z=z, device=model.device, noise=True)
        image = image[0, 0, :, :].detach().cpu().numpy()
        print("min:",image.min().item(), "max:",image.max().item())
        image = image_normalization(image)
        fig = plt.figure()
        plt.axis('off') 
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()
        fig.savefig("generated.png", bbox_inches='tight')
    else:
        #lung_dataset = CTDataset(dataset_path, resolution=2**resolution_power)
        lung_dataset = LungsLabeled(dataset_path, terminate=10, resolution=2**resolution_power, load_memory=False, load_labels=False)
        #print(len(lung_dataset))
        item = np.random.randint(len(lung_dataset))
        print("item:", item)
        image = lung_dataset.__getitem__(item)
        image = image.unsqueeze(0)  
        image = augmentation(image, p_augment=1)
        image_gen, _ = model.autoencoder(x=image, lod=resolution_power - 2, device=device)
        print("min:", image_gen.min().item(), "max:", image_gen.max().item())

        image_gen = image_gen[0,:,:,:].detach().numpy()
        #image_gen = image_normalization(image_gen)

        image = image.squeeze()#.detach().numpy()
        print("---------------------------------------------")
        print('original ct image')
        fig1 = plt.figure(1)
        plt.axis('off') 
        plt.imshow(image[0,:,:], cmap=plt.cm.gray)
        plt.show()
        #fig.savefig("original.png", bbox_inches='tight')
        print('reconstructed ct image')
        fig = plt.figure()
        plt.axis('off') 
        plt.imshow(image_gen[0,:,:], cmap=plt.cm.gray)
        plt.show()
        #fig.savefig("generated.png", bbox_inches='tight')        
        error = np.abs(image[0,:,:] - image_gen[0,:,:]).mean()
        print("ct error:", error)

        print('original pi image')
        fig2 = plt.figure(2)
        plt.axis('off') 
        plt.imshow(image[1,:,:], cmap=plt.cm.gray)
        plt.show()
        print('reconstructed pi image')
        fig = plt.figure()
        plt.axis('off') 
        plt.imshow(image_gen[1,:,:], cmap=plt.cm.gray)
        plt.show()        
        error = np.abs(image[1,:,:] - image_gen[1,:,:]).mean()
        print("pi error:", error)

        # %%
