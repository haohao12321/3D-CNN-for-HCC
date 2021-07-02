import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk
import random

classes = ['0', '1']

def IMG(path,pathp,path_save):
    subfolder = os.listdir(path)
    for nii_file in subfolder:
        img = nib.load(path +'/'+ nii_file).get_fdata()
        img2 = nib.load(pathp+'/'+nii_file.split('.')[0]+'_p.nii').get_fdata()
        img3 = np.multiply(img, img2)
        array_img = nib.Nifti1Image(img3, None)
        nib.save(array_img, path_save+ '/'+nii_file)
        print(nii_file + 'Done')

def CROP(path,path_save):
    for class_name in classes:
        nii_file = os.listdir(path+'/'+class_name)
        for nii in nii_file:
           img = nib.load(path+'/'+class_name+'/'+nii).get_fdata()
           bool_img = img.astype(np.bool)
           axis_list = np.where(bool_img)
           center_x = (axis_list[0].max() + axis_list[0].min()) / 2
           center_y = (axis_list[1].max() + axis_list[1].min()) / 2
           center_z = (axis_list[2].max() + axis_list[2].min()) / 2
           centerpoint = [np.array(center_x, np.int32), np.array(center_y, np.int32),
                          np.array(center_z, np.int32)]
           img_block = img[centerpoint[0] - 75:centerpoint[0] + 75, centerpoint[1] - 75:centerpoint[1] + 75,
                         centerpoint[2] - 5:centerpoint[2] + 5]
           img_block = np.maximum(img_block, 0)
           nib.save(nib.Nifti1Image(img_block, None), path_save+'/'+class_name+'/'+ nii.split('.')[0] + '.nii')

def RANDOM_TRANS(path,path_save):
    img = nib.load(path).get_fdata().astype('float32')
    print(img.shape)
    H, W, C = img.shape
    a = 1
    b = 0
    c = 0
    d = 1
    tx = random.randint(-50,50)
    ty = random.randint(-50,50)
    tem = img.copy()
    img = np.zeros((H + 2, W + 2, C), dtype=np.float32)
    img[1:H + 1, 1:W + 1] = tem
    H_new = np.round(H * d).astype(np.int)
    W_new = np.round(W * a).astype(np.int)
    out = np.zeros((H_new + 1, W_new + 1, C), dtype=np.float32)
    x_new = np.tile(np.arange(W_new), (H_new, 1))
    y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)
    adbc = a * d - b * c
    x = np.round((d * x_new - b * y_new) / adbc).astype(np.int) - tx + 1
    y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1
    x = np.minimum(np.maximum(x, 0), W + 1).astype(np.int)
    y = np.minimum(np.maximum(y, 0), H + 1).astype(np.int)
    out[x_new, y_new] = img[x, y]
    out = out[:H_new, :W_new]
    out = out.astype('float32')
    print(out.shape)
    nib.save(nib.Nifti1Image(out, None),path_save)


def FLIP_qh(path,path_save):
    nii = nib.load(path).get_fdata().astype('float32')
    print(nii.shape)
    res = []
    for i in range(nii.shape[2]):
        i = 9-i
        temp = np.flip(nii[:, :, i],axis=1)
        res.append(temp)
    res = np.array(res)
    res = res.transpose(1, 2, 0)
    new_vol = nib.Nifti1Image(res, None)
    print(new_vol.shape)
    nib.save(new_vol, path_save)

def FLIP_zy(path,path_save):
    nii = nib.load(path).get_fdata().astype('float32')
    print(nii.shape)
    res = []
    for i in range(nii.shape[2]):
        i = 9 - i
        temp = np.flip(nii[:, :, i], axis=0)
        res.append(temp)
    res = np.array(res)
    res = res.transpose(1, 2, 0)
    new_vol = nib.Nifti1Image(res, None)
    print(new_vol.shape)
    nib.save(new_vol, path_save)


def flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr

def ROT(path,path_save):
    nii = nib.load(path).get_fdata().astype('float32')
    print (nii.shape)
    res = []
    for i in range(nii.shape[2]):
        temp = flip90_left(nii[:,:,i])
        res.append(temp)
    res = np.array(res)
    res = res.transpose(1, 2, 0)
    new_vol = nib.Nifti1Image(res, None)
    print(new_vol.shape)
    nib.save(new_vol, path_save)