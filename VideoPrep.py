#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 13:06:16 2018

@author: amolsingh
"""
import cv2
import os
import numpy as np 
#import matplotlib.pyplot as plt
from PIL import Image  # uses pillow
#import time
import imutils, scipy.misc

#%% convert video to set of images
def vid2imgs(path, num):
    #set path for location of video
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    
    #counter for image id
    count = 0
    
    dir_path = os.path.dirname(path)
    newpath = os.path.join(dir_path, "SSOutputFrames%d" %num)
    print(newpath)
    #make new directory for output images
    if not os.path.exists(newpath):
        os.makedirs(newpath)
#        
#    #set working directory to new folder created
    os.chdir(newpath)
    
    #boolean to indicate if there is frame or not
    success = True
    
    #read and create images
    success,image = vidcap.read()
    #saves images in batches of 30 in diff folders
    while success:
        image = resize(image)
#        image = rotateimer(image)
        cv2.imwrite("frame%d.jpg" % count, image)
        count += 1
        success,image = vidcap.read()
        #change to 11 for smaller batch size
        if (count %3 == 0):
            num+=1
            newpath = "/Users/amolsingh/Documents/OhgamiLab/Videos/OutputFrames%d" %num
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            os.chdir(newpath)
#%% reads images and saves them in an array
#def scanImgs(folder):
#    images = []
#    #reads all images in folder
#    for filename in os.listdir(folder):
#        img = cv2.imread(os.path.join(folder,filename))
#        #add images to array
#        if img is not None:            
#            images.append(img)
#    return images
#%%
def scanImgs(folder):
    images = []
    #reads all images in folder
    for filename in os.listdir(folder):
        try:
            img = cv2.imread(os.path.join(folder,filename))
            #for rotating images
#            img = Image.open(os.path.join(folder,filename))
            #add images to array
            if img is not None:            
                images.append(img)
        except:
            print("failed to read")
    return images
#%% algorithm derived from https://github.com/TejasBob/Panorama/blob/master/image-stitching-report.pdf
#location of 
def findFrames(folder):

    images = scanImgs(folder)
    
    index = 0
    #for variance computation
    indexDiff=0
    
    #array of zeros for diffs 
    diff = np.zeros((len(images)+2,1))
    
    #array of key frames
    frames = []
    
    #size of subset of frames used to determine key frame
    batchSize = 10
    
    #loops from first batch to last batch of the images [range([start], stop[, step])]
    for i in range(batchSize,len(images),batchSize):   
       #get dimensions of image fix this!
       imageFile = os.path.join(folder, "frame0.jpg")
       im = Image.open(imageFile)
       (w, h) = im.size
       
       #creates 3D array of zeros of datatype uint8(grayscale)
       gray = np.zeros((h,w,batchSize), dtype = np.uint8)
       #keeps track of index item in array
       index= 0 
       #empty array for storing the mean image from the batch in color
       batchImg = np.zeros((h,w), np.float64)
       #loops through each image of each batch
       for j in range(i-9,i+1,1):
           name = os.path.join(folder ,"frame%d.jpg" % j)
           #loads image in gray scale
           gray[:,:,index] = cv2.imread(name, 0)
           batchImg += gray[:,:,index]
           index+= 1      
       # compute mean image 
       batchImg = batchImg/float(batchSize)
#       plt.imshow(batchImg)
#       plt.show()           
       #makes array of zeros of the batch size
       diff2 = np.zeros((batchSize,1))  
       #compute variance of frames
       for k in range(batchSize):
            diff2[k,0] = np.sum((gray[:,:,k] - batchImg)**2)
       diff2 = diff2/(w*h*1.0)
       diff[indexDiff:(indexDiff+batchSize)] = diff2
       indexDiff+=batchSize  
        # Select frame with minimum variance among batch of frames. Normally it is the middle frame
        # as it has more common region with all other images in the batch
        # Reject frames if all frames are same. That is indicated by zero variance value.
       if np.all(diff2) == True:
            minimum = np.where(diff2 == np.min(diff2))[0][0]
            frames.append(cv2.imread( os.path.join(folder, "frame" + str(i-batchSize + minimum+1) + '.jpg')))
    
    
    return frames
    
#%% find sharpest image in batch
def chooseFrame(folder):
    images = scanImgs(folder)
    #for rotating images
#    intimgs = np.array(imgs)
#    intimgs = (intimgs[:,:,:3] * [0.2989, 0.5870, 0.1140]).sum(axis=2)
#    images = Image.fromarray(intimgs.astype('uint8'), 'RGB')
#    print("working")
    batchSize = 30
    #contains index of global maxes
    maxidxs = []
    frames = []
    
    for i in range(batchSize,len(images),batchSize):   
        #make numpy array of size 30
        sharpval = np.zeros(30)
        for j in range(i-29,i+1,1):
            im = images[j].convert('L') # to grayscale
            #calculate sharpness in terms of average gradient magnitude.
            array = np.asarray(im, dtype=np.int32)
            gy, gx = np.gradient(array)
            gnorm = np.sqrt(gx**2 + gy**2)
            sharpness = np.average(gnorm)            
            #add sharpness value to array    
            np.append(sharpval, sharpness)  
        #return index of max of sharpnesses
        max_idx = np.argmax(sharpval)
        #array of indexes for sharpest images
        maxidxs.append(max_idx)
        
    idxSize = len(maxidxs)
    for k in range(0, idxSize):
        frames.append(images[maxidxs[k]])
        
    return frames
        
        
#%% resize to smaller image for keypointsx
def resize(img):
    basewidth = 400
    h, w, c = img.shape
    wpercent = (basewidth/float(h))
    hsize = int((float(w)*float(wpercent)))
    img = scipy.misc.imresize(img,(basewidth,hsize))
    return img
#%%
def rotateimer(img):
    rotated = imutils.rotate_bound(img, 270)
    return rotated
#%%
#vid2imgs("/Users/amolsingh/Documents/OhgamiLab/Videos/normvid17-12.MOV")

def vid2imgs1vid(path):
    #set path for location of video
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    
    #counter for image id
    count = 0
    num = 1
    
    #comment: make this parameter? **add 1 to end
    newpath = "/Users/amolsingh/Documents/OhgamiLab/Videos/FramesStackNStitch/OutputFrames1"
#    newpath = "/Users/amolsingh/Documents/OhgamiLab/Videos/FramesStackNStitch/OutputFrames(singlebatch)"
    #make new directory for output images
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    #set working directory to new folder created
    os.chdir(newpath)

    #boolean to indicate if there is frame or not
    success = True
    #read and create images
    success,image = vidcap.read()
    #saves images in batches of 30 in diff folders
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)
        count += 1
        success,image = vidcap.read()
        #change to 11 for smaller batch size
        if (count %3 == 0):
            num+=1
            newpath = "/Users/amolsingh/Documents/OhgamiLab/Videos/FramesStackNStitch/OutputFrames%d" %num
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            os.chdir(newpath)