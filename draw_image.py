#%%
import torch
import matplotlib.pyplot as plt

from model import Model
from train import CTDataset

if __name__ == "__main__":
    dataset_path = "/ayb/vol1/kruzhilov/lungs_images/"
    lung_dataset = CTDataset(dataset_path, resolution=2**3)
    device = "cpu"# "cuda:2"
    model = Model(channels=1, device=device)
    model = model.to(device)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()

    z =  torch.randn(6, 128) 
    # image = model.generate(lod=1, blend_factor=1.0, z=z, device=model.device, noise=True)
    # image = image[0, 0, :, :].detach().cpu().numpy()
    # plt.imshow(image, cmap=plt.cm.gray)
    # plt.show()

    image = lung_dataset.__getitem__(3).squeeze(0)
    print('original image')
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()
    # z, _ = model.encode(image, lod=2, blend_factor=1)
    # image_gen = model.generate(lod=2, blend_factor=1, device=device)



# %%
