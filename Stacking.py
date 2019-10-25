#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: amolsingh
"""
import numpy as np
import cv2
#help(cv2.xfeatures2d)
from StackingImagePrep import imageAlignment
from matplotlib import pyplot as plt

#%% reference: http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Gradient_Sobel_Laplacian_Derivatives_Edge_Detection.php
    #http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Gradient_Sobel_Laplacian_Derivatives_Edge_Detection.php
def findLap(image):
            
    #size of a kernal for the gaussian blur
    blur_size = 3
#    plt.imshow(image)
#    plt.show()
    #Perform gaussian blur on all images (LoG filtering) to remove noise
    img = cv2.GaussianBlur(image, (blur_size,blur_size), 2)
#    plt.imshow(img)
#    plt.show()
    #Compute the laplacian on blurred images to make a gradient map for finding in focus regions
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
#    print(lap)
#    plotting the laplacian
#    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
#    plt.title('Original %d' %num), plt.xticks([]), plt.yticks([])
#    plt.subplot(2,2,2),plt.imshow(lap,cmap = 'gray')
#    plt.title('Laplacian %d' %num), plt.xticks([]), plt.yticks([])
#    plt.show()
    return lap

#%%stacks images
def stack(inp):
    #aligns the images
    imgs = imageAlignment(inp)
    laplace = []
    
    #loop through images and compute lap
    for i in range(len(imgs)):
        grayImg = cv2.cvtColor(imgs[i],cv2.COLOR_BGR2GRAY)
        laplace.append(findLap(grayImg))
        
    #converts input to array
    laplace = np.asarray(laplace)
    
    #creates empty array
    out = np.zeros(shape=imgs[0].shape, dtype=imgs[0].dtype)
    
    #find absolute value of laps
    finlaps = np.absolute(laplace)
 
    #for printing size full array
#    np.set_printoptions(threshold=np.nan)
#    i = len(finlaps)
#    x = len(finlaps[1])
#    y = len(finlaps[1][1])
#    print("i " + str(i) + " x " + str(x) + " y " + str(y))    
#    maximum = np.unique()
    
    #find maximum of laps
    maximum = finlaps.max(axis=0) 
    
    #boolean to determine if lap for image is max
    booleanChecker = (finlaps == maximum) 
    
    #pixels are unit8 and uint8 will wrap
    mask = booleanChecker.astype(np.uint8)
    
    #inverts every bit of array using mask that specifies of output array 
    for i in range(0,len(imgs)):
#        out = cv2.bitwise_not(imgs[i],out, mask=mask[i])
        #copies values from one array to another
        #mask is boolean array which specifies which pixel values to copy to 'out'from imgs
        np.copyto(out, imgs[i], where=mask[i][:, :, None].astype(bool))         
#    return 255 - out
    return out
    
    
    
