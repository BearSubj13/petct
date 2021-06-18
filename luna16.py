#%%
import os
import numpy as np
import math

import skimage.io as io
import SimpleITK as sitk
import matplotlib.pyplot as plt


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
    radius_z = math.ceil(radius_z)
    radius_x = 0.5 * radius / spacing[1]
    radius_x_z_list = []
    for r_z in range(-radius_z, radius_z):
        cos_r_z = math.sqrt(1 - (r_z/radius_z)**2 )
        z_i = z + r_z
        radius_x_z = radius_x * cos_r_z / downsize
        if radius_x_z >= 0.5:
            radius_x_z_list.append((z_i, radius_x_z))
    return radius_x_z_list


def heat_map_person(person_nodules, origin, spacing, downsize=1.0):
    z_dict = dict()
    for nodule in person_nodules:
        world_coordinates = (nodule[2], nodule[1], nodule[0])
        voxel = world_2_voxel(world_coordinates, origin, spacing)
        x = round(voxel[2] / downsize)
        y = round(voxel[1] / downsize)
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

    return z_dict


def gaussian_projection(z_i_list, image_size=128):
    image = np.zeros([image_size, image_size])
    for x, y, r in z_i_list:
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

      
if __name__ == "__main__":
    file_path = "/ayb/vol2/datasets/LUNA16/data/"
    #56.20840547,86.34341278,-115.8675792,23.35064438
    file_name = "1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192.mhd"
    #file_name = os.path.join(file_path, file_name)

    nodule_dict = read_annotation(file_path="/ayb/vol2/datasets/LUNA16/CSVFILES/annotations.csv")

    fl = False
    for human_id in nodule_dict.keys():
        if len(nodule_dict[human_id]) > 4:
            for nodule in nodule_dict[human_id]:
                #if nodule[3] > 3.0 and nodule[3] < 3.5:
                world_coordinates = (nodule[2], nodule[1], nodule[0])
                file_name = human_id + ".mhd"
                file_name = os.path.join(file_path, file_name)
                img, origin, spacing = load_itk(file_name)
                voxel = world_2_voxel(world_coordinates, origin, spacing)
                radius = radius_2_voxel(nodule[3], spacing)
                radius_list = radius_nodule(radius, round(voxel[0]), spacing)
                z_dict = heat_map_person(nodule_dict[human_id], origin, spacing, downsize=4)
                len_list = [True for key in z_dict if len(z_dict[key])>=2]
                if len_list:
                    fl = True
                    break
            if fl:
                break               

    z = 69
    z_dict = heat_map_person(nodule_dict[human_id], origin, spacing, downsize=4)
    image = gaussian_projection(z_dict[z], 128)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()

    fig, ax = plt.subplots()
    for x, y, r in z_dict[z]:
        circle = plt.Circle((x, y), 1.3*r, color='r', fill=False)
        image = img[z,:,:]   
        image = image[0:512:4, 0:512:4]
        plt.imshow(image, cmap=plt.cm.gray)
        ax.add_artist(circle)

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
