# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 17:19:32 2018

@author: Administrator
"""

import nibabel as nib
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from central_pooling import central_pooling,central_pooling_out_shape
from keras.layers.core import Lambda
import keras
from keras.initializers import Constant
def get_model(inputs=(35,35,2)):
    model = Sequential()
    model.add(Conv2D(36, (3, 3), padding = 'same', kernel_initializer = 'glorot_normal',input_shape=inputs))
    model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zeros'))
    model.add(Conv2D(36, (3, 3), padding = 'same', kernel_initializer = Constant(0.25)))
    model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zeros'))
    model.add( Lambda(central_pooling))
    model.add(Conv2D(48, (3, 3),  padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(keras.layers.advanced_activations.PReLU(alpha_initializer=Constant(0.25)))
    model.add(Conv2D(48, (3, 3),   padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(keras.layers.advanced_activations.PReLU(alpha_initializer=Constant(0.25)))
    model.add( Lambda(central_pooling))
    model.add(Conv2D(68, (3, 3),   padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(keras.layers.advanced_activations.PReLU(alpha_initializer=Constant(0.25)))
    model.add(Conv2D(68, (3, 3),   padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(keras.layers.advanced_activations.PReLU(alpha_initializer=Constant(0.25)))
    model.add(Flatten())
    model.add(Dense(300,kernel_initializer = 'glorot_normal'))
    model.add(keras.layers.advanced_activations.PReLU(alpha_initializer=Constant(0.25)))
    model.add(Dense(2,activation ='softmax',kernel_initializer = 'glorot_normal'))
    model.compile(optimizer=SGD(lr=6.0e-5),loss='binary_crossentropy')
    return model

if __name__=='__main__':
    
    a=np.random.rand(2,35,35,2)
    print(a.shape)
    b=np.array([[1,0],
                [0.5,0.5]])
    print(b.shape)
    model=get_model()
    model_checkpoint = ModelCheckpoint('2.hdf5',monitor='loss', save_best_only=True)
    model.fit(a, b, batch_size=1, nb_epoch=1, verbose=1, shuffle=True,
              callbacks=[model_checkpoint])
        
    
    
    
    