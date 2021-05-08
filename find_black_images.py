#%%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    folder_path = "/ayb/vol1/kruzhilov/lungs_images_val/"
    file_list = []
    i = 0
    for file_name in os.listdir(folder_path):
        fname = os.path.join(folder_path, file_name)
        image_tensor = torch.load(fname)

        # image_tensor = image_tensor.unsqueeze(0)#.unsqueeze(0)
        # image_tensor = F.interpolate(image_tensor, size=128)
        # image_tensor = image_tensor.permute(0, 2, 1)
        # image_tensor = F.interpolate(image_tensor, size=128 ) 
        # image_tensor = image_tensor.permute(0, 2, 1)
        # image_tensor = image_tensor.squeeze()
        # torch.save(image_tensor, fname)

        # im_sum =  image_tensor.sum()
        # if im_sum < 1800:
        #     file_list.append(file_name)
        #     print(im_sum)
        i = i + 1
        if i == 200:
            plt.imshow(image_tensor, cmap=plt.cm.gray)
            plt.show()
            break
    
    # for file_name in file_list:
    #     fname = os.path.join(folder_path, file_name)
    #     os.remove(fname)


# %%
