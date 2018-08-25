# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:04:46 2018

@author: Administrator
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 10:44:22 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 20:48:24 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 10:44:22 2018

@author: Administrator
"""

import nibabel as nib
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import tensorflow as tf
import copy
#tf.reset_default_graph()
TABEL=np.load('lookup_table.npy')


   
def central_pooling(inputs):
    size=[1,2,3]
    input_size=int(inputs.get_shape()[1])
    
    n1=math.floor(input_size/8)
    
    n2=math.floor(input_size/4)
    n3=math.floor(input_size/8)
    residual=input_size-n1*1-n2*2-n3*3
    L=look_up(residual)
    
    n1=n1+L[1]
    n2=n2+L[2]
    n3=n3+L[3]
    out=[]
    assert(n1+2*n2+3*n3==input_size)
    n3_up=inputs[:,:3*math.ceil(n3/2),:,:]
    n3_up_out1=tf.nn.max_pool(n3_up,[1,3,1,1],[1,3,1,1],padding='VALID')
    out.append(n3_up_out1)
    n2_up=inputs[:,3*math.ceil(n3/2):3*math.ceil(n3/2)+2*math.ceil(n2/2),:,:]
    n2_up_out1=tf.nn.max_pool(n2_up,[1,2,1,1],[1,2,1,1],padding='VALID')
    n1_up=inputs[:,3*math.ceil(n3/2)+2*math.ceil(n2/2):3*math.ceil(n3/2)+2*math.ceil(n2/2)+math.ceil(n1/2),:,:]
    out.append(n2_up_out1)
    n1_up_out1=tf.nn.max_pool(n1_up,[1,1,1,1],[1,1,1,1],padding='VALID')
    out.append(n1_up_out1)
    index=3*math.ceil(n3/2)+2*math.ceil(n2/2)+math.ceil(n1/2)
    if((n1-math.ceil(n1/2))>0):
        n1_bottom=inputs[:,index:int(index+(n1-math.ceil(n1/2))),:,:]
        index=int(index+(n1-math.ceil(n1/2)))
        n1_bottom_out=tf.nn.max_pool(n1_bottom,[1,1,1,1],[1,1,1,1],padding='VALID')
        out.append(n1_bottom_out)
    if((n2-math.ceil(n2/2))>0):
        n2_bottom=inputs[:,index:int(index+2*(n2-math.ceil(n2/2))),:,:]
        index=int(index+2*(n2-math.ceil(n2/2)))
        n2_bottom_out=tf.nn.max_pool(n2_bottom,[1,2,1,1],[1,2,1,1],padding='VALID')
        out.append(n2_bottom_out)
    if((n3-math.ceil(n3/2))>0):
        
        n3_bottom=inputs[:,index:int(index+3*(n3-math.ceil(n3/2))),:,:]
        index=int(index+3*(n3-math.ceil(n3/2)))
        n3_bottom_out=tf.nn.max_pool(n3_bottom,[1,3,1,1],[1,3,1,1],padding='VALID')
        out.append(n3_bottom_out)

    concat=tf.concat(out,axis=1)
    out1=[]
    n3_left=concat[:,:,:3*math.ceil(n3/2),:]
    n3_left_out=tf.nn.max_pool(n3_left,[1,1,3,1],[1,1,3,1],padding='VALID')
    out1.append(n3_left_out)
    n2_left=concat[:,:,3*math.ceil(n3/2):3*math.ceil(n3/2)+2*math.ceil(n2/2),:]
    n2_left_out=tf.nn.max_pool(n2_left,[1,1,2,1],[1,1,2,1],padding='VALID')
    out1.append(n2_left_out)
    n1_left=concat[:,:,3*math.ceil(n3/2)+2*math.ceil(n2/2):3*math.ceil(n3/2)+2*math.ceil(n2/2)+math.ceil(n1/2),:]
    n1_left_out=tf.nn.max_pool(n1_left,[1,1,1,1],[1,1,1,1],padding='VALID')
    out1.append(n1_left_out)
    index=3*math.ceil(n3/2)+2*math.ceil(n2/2)+math.ceil(n1/2)
    if((n1-math.ceil(n1/2))>0):
        n1_right=concat[:,:,index:int(index+(n1-math.ceil(n1/2))),:]
        
        index=int(index+(n1-math.ceil(n1/2)))
        n1_right_out=tf.nn.max_pool(n1_right,[1,1,1,1],[1,1,1,1],padding='VALID')
        out1.append( n1_right_out)
    if((n2-math.ceil(n2/2))>0):
        n2_right=concat[:,:,index:int(index+2*(n2-math.ceil(n2/2))),:]
        
        index=int(index+2*(n2-math.ceil(n2/2)))
        n2_right_out=tf.nn.max_pool(n2_right,[1,1,2,1],[1,1,2,1],padding='VALID')
       
        out1.append(n2_right_out)
    if((n3-math.ceil(n3/2))>0):
        
        
        n3_right=concat[:,:,index:int(index+3*(n3-math.ceil(n3/2))),:]
        
        index=int(index+3*(n3-math.ceil(n3/2)))
        n3_right_out=tf.nn.max_pool(n3_right,[1,1,3,1],[1,1,3,1],padding='VALID')
        out1.append( n3_right_out)
    concat1=tf.concat(out1,axis=2)
    
        
    return  concat1
        
def  central_pooling_out_shape(input_shape):
    shape = list(input_shape)
    for i in shape:
        i=math.ceil(i/2)
    return tuple(shape)
   
    
        
        
    
    

def look_up(r):
    return TABEL[:,r]
    
        
if __name__ == '__main__':
    a=np.random.rand(9,9,3)[np.newaxis,:,:,:]
    
    
    a=tf.convert_to_tensor(a)
    
    
    c=central_pooling(a)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(c).shape)
    


