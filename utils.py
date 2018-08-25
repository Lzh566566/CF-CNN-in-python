# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:17:29 2018

@author: Administrator
"""

import skimage.io as io
from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt
import cv2
def patch_index(mask,img):
    index_1=np.where(mask==1)
    
    index_0=np.where(mask==0)
    
    nodule=np.concatenate((index_1[0][:,np.newaxis],index_1[1][:,np.newaxis]),axis=1)
   
    none_nodule=np.concatenate((index_0[0][:,np.newaxis],index_0[1][:,np.newaxis]),axis=1)
    dis=spatial.distance.cdist(nodule,none_nodule,'euclidean')
    

    
    dis_pw=np.min(dis,axis=1)
    
    sorted_index=np.argsort(dis_pw)
    
    nodule_index=nodule[sorted_index]
    

    dis_nw=np.min(dis.T,axis=1)
    a=np.exp(-dis_nw)
    
    for dis_i,pixel in zip(a,none_nodule):
        dis_i=dis_i*img[pixel]
    sorted_index=np.argsort(a)[::-1]
    
    none_nodule_index=none_nodule[sorted_index]
    return nodule_index,none_nodule_index
    
def get_patch(img,index,width=[35,65],height=[35,65]):
    
    lis=[]
    for i,j in index:
        patch0=img[i-width[0]//2:i+width[0]-width[0]//2,j-height[0]//2:j+height[0]-height[0]//2]
        patch1=img[i-width[1]//2:i+width[1]-width[1]//2,j-height[1]//2:j+height[1]-height[1]//2]
        if ((patch0.shape!=(width[0],height[0])or(patch1.shape!=(width[1],height[1])))):
            continue
        
        patch1=cv2.resize(patch1,(width[0],height[0]),interpolation=cv2.INTER_CUBIC)
#        plt.imshow(patch0)
#        plt.show()
#        plt.imshow(patch1)
#        plt.show()
        patch=np.concatenate((patch0[:,:,np.newaxis],patch1[:,:,np.newaxis]),axis=-1)
        lis.append(patch)
    lis=np.array(lis)
    
    return lis
           
        
def reflect_padding(img,padding_size):
    '''
    img:ndarray
    '''
    row,col=img.shape
    assert row==col
    assert padding_size<=row
    new_row=new_col=row+2*padding_size
    mirror_block_up=img[0:padding_size,:][::-1,:]
    
    mirror_block_diagonal1=mirror_block_up[:,:padding_size][:,::-1]
    
    mirror_block_diagonal2=mirror_block_up[:,-padding_size:][:,::-1]
    
    up=np.concatenate((mirror_block_diagonal1,mirror_block_up,mirror_block_diagonal2),axis=1)
    
    mirror_block_bottom=img[-padding_size:,:][::-1,:]
    
    mirror_block_diagonal3=mirror_block_bottom[:,:padding_size][:,::-1]
    mirror_block_diagonal4=mirror_block_bottom[:,-padding_size:][:,::-1]
    bottom=np.concatenate( (mirror_block_diagonal3,mirror_block_bottom,mirror_block_diagonal4),axis=1)
    
    mirror_block_left=img[:,0:padding_size][:,::-1]
    print(mirror_block_left.shape)
    mirror_block_rjght=img[:,-padding_size:][:,::-1]
    middle=np.concatenate((mirror_block_left,img,mirror_block_rjght),axis=1)
    print(middle.shape)
    return np.concatenate((up,middle,bottom),axis=0)
if __name__ == '__main__':
    a=io.imread(r"C:\Users\Administrator\Desktop\CF-CNN\img.png") 
    b=reflect_padding(a,20)
    plt.imshow(b)
    plt.show()
    
    
    
    
    
    

    
    
    
    
        
    