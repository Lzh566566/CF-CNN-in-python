# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 16:42:32 2018

@author: Administrator
"""

from model import *
from utils import patch_index,get_patch
import cv2
import numpy as np
import keras.utils
img=cv2.imread('img.png',-1)
mask=cv2.imread('mask.png',-1)

nodule_index,none_nodule_index=patch_index(mask,img)
nodule_patch=get_patch(img,nodule_index)
mask_1=np.ones([nodule_patch.shape[0],1])
mask_1=keras.utils.np_utils.to_categorical(mask_1,num_classes=2)
print(mask_1)

none_nodule_patch=get_patch(img,none_nodule_index)
mask_0=np.zeros([none_nodule_patch.shape[0],1])
mask_0=keras.utils.np_utils.to_categorical(mask_0,num_classes=2)
print(mask_0)
train_data=np.concatenate((nodule_patch,none_nodule_patch),axis=0)
mean=np.mean(train_data)
std=np.std(train_data)
train_data=(train_data-mean)/std
mask=np.concatenate((mask_1,mask_0),axis=0)

model=get_model()
model_checkpoint = ModelCheckpoint('2.hdf5',monitor='loss', save_best_only=True)
model.fit(train_data, mask, batch_size=2, nb_epoch=2, verbose=1, shuffle=True,
          callbacks=[model_checkpoint])