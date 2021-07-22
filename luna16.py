#%%
import os
import numpy as np
import math
import random
from shutil import rmtree

import skimage.io as io
import skimage
import SimpleITK as sitk
import matplotlib.pyplot as plt

from scipy.ndimage import rotate
from skimage.transform import resize
from torch.utils.data import Dataset
import torch


def random_crop(img1, img2, crop_size=(10, 10)):
    assert crop_size[0] <= img1.shape[0] and crop_size[1] <= img1.shape[1], "Crop size should be less than image size"
    img1 = img1.copy()
    img2 = img2.copy()
    w, h = img1.shape[:2]
    x, y = np.random.randint(h-crop_size[0]), np.random.randint(w-crop_size[1])
    img1 = img1[y:y+crop_size[0], x:x+crop_size[1]]
    img1 = resize(img1, (w,h))
    img2 = img2[y:y+crop_size[0], x:x+crop_size[1]]
    img2 = resize(img2, (w,h))
    return img1, img2


class Luna_heat_map_dataset(Dataset):
    """
    """
    def __init__(self, dataset_path, file_list=None, terminate=False, transform=False):
        self.transform = transform
        if file_list is None:
            file_list = sorted(os.listdir(dataset_path))
        self.image = []
        self.heatmap = []
        self.coordinates = []
        for id, human_id in enumerate(file_list):
            if terminate and id > terminate:
                break
            file_path = os.path.join(dataset_path, human_id)
            variable_dict = np.load(file_path, allow_pickle=True)
            for i in range(variable_dict["img"].shape[0]):
                self.image.append(variable_dict["img"][i,:,:])
                self.heatmap.append(variable_dict["heat_map"][i,:,:])
                coord = variable_dict["xyr"][i]
                coord = np.array(coord)
                coord = coord.reshape(np.prod(coord.shape))
                self.coordinates.append(coord)

    def __getitem__(self, index):
        image = self.image[index]
        heatmap = self.heatmap[index]
        if self.transform:
            random_coin = np.random.rand()
            if random_coin < 0.25:
                image = np.fliplr(image)
                heatmap = np.fliplr(heatmap)
            elif random_coin >= 0.25 and random_coin <= 0.5:
               image, heatmap = random_crop(image, heatmap, crop_size=(110, 110))
            elif random_coin > 0.5 and random_coin <= 0.65:
                angle = 90 * (np.random.rand() - 1)
                image = rotate(image, angle, reshape=False)
                heatmap = rotate(heatmap, angle, reshape=False)
            image = image.copy()
            heatmap = heatmap.copy()
            return image, heatmap, torch.FloatTensor([0])
        else:
            coordinates = self.coordinates[index]
            coordinates = coordinates.astype(np.float32)
            coordinates = torch.FloatTensor(coordinates)
            return image, heatmap, coordinates

    def __len__(self):
        return len(self.image)


def train_val_split_luna(dataset_path, val_rate=0.1):
    files_list = sorted(os.listdir(dataset_path))
    val_len = round(val_rate * len(files_list))
    random.seed(666)
    val_list = random.sample(files_list, val_len)
    train_list = list(set(files_list) - set(val_list))
    train_list = sorted(train_list)
    train_dataset = Luna_heat_map_dataset(dataset_path, train_list, transform=True)
    val_dataset = Luna_heat_map_dataset(dataset_path, val_list)
    return train_dataset, val_dataset



'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, 
origin and spacing of the image.
'''
def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    
    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    
    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    
    return ct_scan, origin, spacing


'''
This function is used to convert the world coordinates to voxel coordinates using 
the origin and spacing of the ct_scan
'''
def world_2_voxel(world_coordinates, origin, spacing, downsize=1.0):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    voxel_coordinates[2] = round(voxel_coordinates[2]/downsize)
    voxel_coordinates[1] = round(voxel_coordinates[1]/downsize)
    return voxel_coordinates


'''
This function is used to convert the voxel coordinates to world coordinates using 
the origin and spacing of the ct_scan.
'''
def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates


def radius_2_voxel(radius, spacing):
    return 0.5 * radius / spacing[1]


def read_annotation(file_path):
    with open(file_path) as f:
        lines = [line.rstrip() for line in f]
    del lines[0]

    nodule_dict = dict()
    for line in lines:
        id_coord = line.split(",")
        id = id_coord[0]
        x = float(id_coord[1])
        y = float(id_coord[2])
        z = float(id_coord[3])
        r = float(id_coord[4])
        if id not in nodule_dict.keys():
            nodule_dict.update({id:[(x, y, z, r)]})
        else:
            nodule_dict[id].append((x, y, z, r))

    return nodule_dict


def radius_nodule(radius, z, spacing, downsize=1.0):
    radius_z =  0.5 * radius / spacing[0]
    radius_z = math.floor(1.3*radius_z) + 1
    radius_x = 0.5 * radius / spacing[1]
    radius_x_z_list = []
    for r_z in range(-radius_z, radius_z):
        cos_r_z = math.sqrt(1 - (r_z/radius_z)**2 )
        z_i = z + r_z
        radius_x_z = radius_x * cos_r_z / downsize
        if radius_x_z >= 0.6:
            radius_x_z_list.append((z_i, radius_x_z))
    return radius_x_z_list


def d2points_with_radius(person_nodules, origin, spacing, downsize=1.0):
    z_dict = dict()
    original_coord_list = []
    for nodule in person_nodules:
        world_coordinates = (nodule[2], nodule[1], nodule[0])
        voxel = world_2_voxel(world_coordinates, origin, spacing)
        x = voxel[2] / downsize
        y = voxel[1] / downsize
        z = round(voxel[0])
        r = radius_2_voxel(nodule[3], spacing) / downsize
        radius_list = radius_nodule(r, z, spacing)
        for z_r in radius_list:
            z_i = z_r[0]  
            r_i = z_r[1]
            if z_i in z_dict.keys():
                z_dict[z_i].append((x, y, r_i))
            else:
                z_dict.update({z_i:[(x, y, r_i)]})

    return z_dict #, original_coord_list


def gaussian_projection(z_i_list, image_size=128, radius_coeff=1.0):
    image = np.zeros([image_size, image_size])
    for x, y, r in z_i_list:
        if r < 2.5:
            r = 2.5
        r = 5.0 if r > 5.0 else r
        r = radius_coeff * r #to increase the area of a heatmap
        x_begin = round(x - 3*r)
        x_end = round(x + 3*r)
        y_begin = round(y - 3*r)
        y_end = round(y + 3*r)
        for i in range(x_begin, x_end):
            for j in range(y_begin, y_end):
                r_from_center = np.linalg.norm(np.array([x, y]) - np.array([i, j]))
                coeff = math.exp(-0.5*(r_from_center / r)**2)
                if coeff > 0.05:
                    image[j, i] = coeff

    return image


def tensor_for_person(file_path, human_id, nodule_dict, downsize=1):
    image_size = 512 // downsize
    file_name = human_id + ".mhd"
    file_name = os.path.join(file_path, file_name)
    img, origin, spacing = load_itk(file_name)
    z_dict = d2points_with_radius(nodule_dict[human_id], origin, spacing, downsize=downsize)
    if len(z_dict) == 0:
        return None, None, None
    z_numpy = np.array(sorted(z_dict.keys()))
    heat_map = np.zeros([z_numpy.shape[0], image_size, image_size])
    x_y_r_list = []
    for i, z in enumerate(z_numpy):
        heat_map[i,:,:] = gaussian_projection(z_dict[z], image_size=image_size, radius_coeff=1.1)
        x_y_r_list.append(z_dict[z])
    img = img[z_numpy, :,:]
    img = img[:, 0:512:downsize, 0:512:downsize]
    img = np.array(img, dtype=np.float64)
    if img.max() - img.min() < 0.000001:
        return None, None, None
    img = (img - img.min()) / (img.max() - img.min())
    assert img.shape == heat_map.shape
    return img, heat_map, x_y_r_list


def find_stars(heatmap, threshold=0.1, min_energy_mass=1.0):
    heatmap[heatmap < threshold] = 0
    image_size = heatmap.shape[0]
    heatmap_nonempty = heatmap.copy()
    heatmap_nonempty[heatmap_nonempty > threshold] = 1
    blobs = skimage.measure.label(heatmap_nonempty, return_num=False)
    number_of_blobs = blobs.max()
    xyr = np.zeros([number_of_blobs, 3])
    k = 0
    for i in range(1, number_of_blobs+1):
        mask = blobs == i
        coordinates = np.where(mask)
        energy = heatmap[coordinates[0], coordinates[1]]
        energy_mass = energy.sum()
        if energy_mass < min_energy_mass:
            continue
        y_engergy = (energy * coordinates[0]).sum()
        x_engergy = (energy * coordinates[1]).sum()
        x_estim = y_engergy / energy_mass
        y_estim = x_engergy / energy_mass
        xy_repeat = np.repeat(np.array([[x_estim, y_estim]]), repeats=energy.shape[0], axis=0)
        xy_repeat = xy_repeat.transpose()
        coordinates = np.vstack(coordinates)
        distorsion = (energy/energy_mass)*(xy_repeat - coordinates)**2
        r = np.sqrt(distorsion.sum())
        xyr[k, 0] = y_estim
        xyr[k, 1] = x_estim
        xyr[k, 2] = r
        k = k + 1
    xyr = xyr[:k, :]
    return xyr


def coordinates_metrics(xy_estim, xyr_gt, efficient_radius=1.0):
    pair_list = []
    mean_err = []
    for i in range(xyr_gt.shape[0]):
        r = xyr_gt[i,2]
        xy = xyr_gt[i,:2]
        for j in range(xy_estim.shape[0]):
            d = np.linalg.norm(xy - xy_estim[j,:2])
            if d < efficient_radius * r:
                pair_list.append((i,j))
                mean_err.append(d)
                break

    recall = len(pair_list) / xyr_gt.shape[0]
    if xy_estim.shape[0] == 0:
        precision = None
    else:
        precision = len(pair_list) / xy_estim.shape[0]
    if len(mean_err) > 0:
        mean_err = sum(mean_err) / len(mean_err)
    else:
        mean_err = None
    return precision, recall, mean_err

      
if __name__ == "__main__":
    file_path = "/ayb/vol2/datasets/LUNA16/data/"
    save_path = "/ayb/vol1/kruzhilov/datasets/luna16_heatmap/resolution256_largeheatmap"

    # train_dataset, val_dataset = train_val_split_luna(save_path, val_rate=0.1)
    # print("train len:", len(train_dataset), ", val len:", len(val_dataset))
    
    # k = 500
    # print(k)
    # image, heat_map, xyr = train_dataset.__getitem__(k)
    # fig, ax = plt.subplots()
    # plt.imshow(image, cmap=plt.cm.gray)
    # for x, y, r in xyr:
    #     circle = plt.Circle((x, y), 1.3*r, color='r', fill=False)
    #     ax.add_artist(circle)
    # plt.show()
    # plt.imshow(heat_map, cmap=plt.cm.gray)
    # plt.show()

    if os.path.exists(save_path):
        rmtree(save_path)
    os.mkdir(save_path)

    nodule_dict = read_annotation(file_path="/ayb/vol2/datasets/LUNA16/CSVFILES/annotations.csv")

    for i, human_id in enumerate(nodule_dict.keys()):
        img, heat_maps, x_y_r_list = tensor_for_person(file_path, human_id, nodule_dict, downsize=2)
        file_to_save = human_id + ".npz"
        file_to_save = os.path.join(save_path, file_to_save)
        x_y_r_list = np.array(x_y_r_list, dtype=object)
        if img is not None and heat_maps is not None:
            np.savez(file_to_save, img=img, heat_map=heat_maps, xyr=x_y_r_list)
        # if i == 50:
        #     break

        
    # fl = False #i==20, 32 - None
    # for i, human_id in enumerate(nodule_dict.keys()):
    #     if i == 43:#len(nodule_dict[human_id]) >= 2:
    #         #for nodule in nodule_dict[human_id]:
    #             #if nodule[3] > 3.0 and nodule[3] < 3.5:
    #             # world_coordinates = (nodule[2], nodule[1], nodule[0])
    #             # file_name = human_id + ".mhd"
    #             # file_name = os.path.join(file_path, file_name)
    #             # img, origin, spacing = load_itk(file_name)
    #             # voxel = world_2_voxel(world_coordinates, origin, spacing)
    #             # radius = radius_2_voxel(nodule[3], spacing)
    #             # radius_list = radius_nodule(radius, round(voxel[0]), spacing)
    #         img, heat_maps, z_dict = tensor_for_person(human_id, nodule_dict, downsize=4)
    #             # len_list = [True for key in z_dict if len(z_dict[key])>=2]
    #             # if len_list:
    #      #       fl = True
    #         if img is None:
    #             break
    #         else:
    #             break
    #     #if fl:
    #     #        break               

    # sorted_keys = sorted(z_dict.keys())
    # for i, z in enumerate(sorted_keys):
    #     plt.imshow(heat_maps[i], cmap=plt.cm.gray)
    #     plt.show()

    #     fig, ax = plt.subplots()
    #     image = img[i,:,:]   
    #     #image = image[0:512:4, 0:512:4]
    #     plt.imshow(image, cmap=plt.cm.gray)
    #     for x, y, r in z_dict[z]:
    #         circle = plt.Circle((x, y), 1.2*r, color='r', fill=False)
    #         image = img[i,:,:]   
    #         #image = image[0:512:4, 0:512:4]
    #         plt.imshow(image, cmap=plt.cm.gray)
    #         ax.add_artist(circle)
    #     plt.show()
    #     print(" ")

    # for z, r in radius_list:
    #     downsize = 4.0
    #     r = r / downsize
    #     z = round(z)
    #     x = round(voxel[2]/downsize)
    #     y = round(voxel[1]/downsize)
    #     circle = plt.Circle((x, y), 1.3*r, color='r', fill=False)
    #     fig, ax = plt.subplots()    
    #     image = img[z,:,:]   
    #     image = image[0:512:4, 0:512:4] 
    #     print(image.shape)
    #     plt.imshow(image, cmap=plt.cm.gray)
    #     #ax = plt.gca()
    #     ax.set_aspect(1)
    #     ax.add_artist(circle)
    #     #plt.imshow(img[z,:,:], cmap=plt.cm.gray)
    # plt.show()
        

    print("the end")
# %%
