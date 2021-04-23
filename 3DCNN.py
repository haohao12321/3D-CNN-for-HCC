from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv3D,AveragePooling3D,MaxPooling3D
#from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.optimizers import Adam,SGD
from keras.utils import np_utils
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
import nibabel as nib
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold

def X_TRAIN(train_folder):
    X_tr = []
    # Reading data from each class
    for class_name in classes:
        class_folder = os.path.join(train_folder, class_name)  # 连接路径的函数
        listing = os.listdir(class_folder)
        for patient in listing:
            patient_nii = nib.load(train_folder + '/' + class_name + '/' + patient).get_fdata().astype('float16')
            patient_narray = np.array(patient_nii)
            patient_narray = patient_narray.transpose(2, 0, 1)
            X_tr.append(patient_narray)
    x_train = np.array(X_tr)
    print(x_train.shape)# (, z, x, y)
    x_train = np.expand_dims(x_train, axis=-1)    # 加一维
    x_train = x_train.astype('float16')  # 16浮点数
    print(np.max(x_train))
    #x_train = x_train / np.max(x_train) # 归一化  400
    x_train = x_train / 400  # 归一化
    print(x_train.shape)
    return x_train

def Y_TRAIN(x_train):
    num_samples = x_train.shape[0]
    label = np.ones((num_samples,), dtype=int)
    label[0:735] = 0  # 0
    label[735:1246] = 1  # 1
    y_train = label
    # convert class vectors to binary class matrices将类向量转换为二进制类矩阵
    return  y_train


def CNN_3D():
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



def MODEL_FIT(model,x_train,y_train,batch_size,nb_epoch):
    optimizer = Adam(lr=0.00001)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=30)
    weights = {0: 0.84761905, 1: 1.21917808}
    index = 1

    for train, test in kfold.split(x_train, y_train):
        filepath = 'F:/model/model_{epoch:03d}-{val_accuracy:.4f}-{val_loss:.4f}_' + str(index) + '.h5'
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        checkpoint = ModelCheckpoint(filepath, verbose=1,
                                     save_best_only=False, save_weights_only=True, mode='auto', period=1)
        callbacks_list = [checkpoint]
        model.fit(x_train[train], np_utils.to_categorical(y_train[train], 2), class_weight=weights,
                  callbacks=callbacks_list,epochs=nb_epoch, batch_size=batch_size,
                  validation_data=(x_train[test], np_utils.to_categorical(y_train[test])),shuffle=True)
        scores = model.evaluate(x=x_train[test], y = np_utils.to_categorical(y_train[test], 2), verbose=0)
        cvscores.append(scores[1] * 100)
        # train_data, train_label = prepare_data()
        # model.fit(train_data, train_label, batch_size=64, epochs=20, shuffle=True, validation_split=0.2)
        index = index + 1


cvscores = []
classes = ['0', '1']
batch_size = 12
nb_epoch =20
train_folder = 'C:/Users/cuihao/Desktop/train'
x_train = X_TRAIN(train_folder)
y_train = Y_TRAIN(x_train)
model = CNN_3D()
MODEL_FIT(model,x_train,y_train,batch_size,nb_epoch)
