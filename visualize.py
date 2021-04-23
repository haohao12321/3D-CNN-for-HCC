from vis.utils import utils
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv3D,AveragePooling3D,MaxPooling3D
from keras.models import Sequential
import nibabel as nib



def LETNET4():
    model = Sequential()
    model.add(Conv3D(32, kernel_size= (3, 3, 3), strides=(1, 1, 1),activation='relu',padding='same', input_shape=(10, 150, 150, 1)))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same',strides=(2, 2, 2)))
    model.add(Conv3D(64, kernel_size = (3, 3, 3),strides=(1, 1, 1),padding='same',activation='relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same',strides=(2, 2, 2)))
    model.add(Conv3D(128, kernel_size = (3, 3, 3),strides=(1, 1, 1),padding='same',activation='relu'))
    model.add(Conv3D(128, kernel_size = (3, 3, 3),strides=(1, 1, 1),padding='same',activation='relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same',strides=(2, 2, 2)))
    model.add(Conv3D(256, kernel_size = (2, 3, 3),strides=(1, 1, 1),padding='same',activation='relu'))
    model.add(Conv3D(256, kernel_size = (2, 3, 3),strides=(1, 1, 1),padding='same',activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 3, 3), padding='same',strides=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.summary()  # 输出模型参数
    return model

def model():
    model = LETNET4()
    model.load_weights("F:/model/bj.h5")
    model.summary()
    return model

import numpy as np
from vis.visualization import visualize_cam,overlay

img = nib.load("C:/Users/cuihao/Desktop/0005428.nii").get_fdata().astype('float16')

img = np.array(img)
img = img/400
img = img.transpose(2, 0, 1)
img = np.expand_dims(img, axis=-1)  # 加一维
img = np.expand_dims(img, axis=0)
print(img.shape)
modifier=[None,'guided','relu']

def cam(model):
   layer_idx=utils.find_layer_idx(model=model,layer_name='dense_4')
   grad = visualize_cam(model, layer_idx,filter_indices=1,seed_input=img, backprop_modifier='guided')
   plt.imshow(grad[0, :, :], cmap='jet')
   plt.show()
   plt.imshow(grad[1, :, :], cmap='jet')
   plt.show()
   plt.imshow(grad[2, :, :], cmap='jet')
   plt.show()
   plt.imshow(grad[3, :, :], cmap='jet')
   plt.show()
   plt.imshow(grad[4, :, :], cmap='jet')
   plt.show()
   plt.imshow(grad[5, :, :], cmap='jet')
   plt.show()
   plt.imshow(grad[6, :, :], cmap='jet')
   plt.show()
   plt.imshow(grad[7, :, :], cmap='jet')
   plt.show()
   plt.imshow(grad[8, :, :], cmap='jet')
   plt.show()
   plt.imshow(grad[9, :, :], cmap='jet')
   plt.show()


model = model()
cam(model)