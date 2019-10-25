#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 15:33:10 2018

@author: amolsingh
"""
import cv2
#help(cv2.xfeatures2d)
import os, time
import sys
from PIL import Image
from Stacking import stack
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#import glob

#%%
def stacker(folder):
#    time1 = time.time()
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    stacked = stack(images)
#    time2 = time.time()
#    fintime = time2 - time1
#    print("Time: " + str(fintime))
    return stacked
#    newpath = folder
#    os.chdir(newpath)
#    cv2.imwrite("Stacked.png", stacked)

def stacker1(imgList):
    time1 = time.time()
    images = []
    for imageFile in imgList:
        img = cv2.imread(imageFile)
        if img is not None:
            images.append(img)
    stacked = stack(images)
    time2 = time.time()
    fintime = time2 - time1
    print("Time: " + str(fintime))
    return stacked
#
#imgs = ['/Users/amolsingh/Documents/OhgamiLab/Images/Stacking Images/Q1.png', '/Users/amolsingh/Documents/OhgamiLab/Images/Stacking Images/Q1a.png', '/Users/amolsingh/Documents/OhgamiLab/Images/Stacking Images/Q1b.png']
#img = stacker1(imgs)
#cv2.imwrite('/Users/amolsingh/Documents/OhgamiLab/Images/Stacking Images/Stackedarray.png', img)
#%% 

#img = stacker("/Users/amolsingh/Documents/OhgamiLab/Videos/FramesStackNStitch/OutputFrames28")
#cv2.imwrite("/Users/amolsingh/Documents/OhgamiLab/Videos/Stackedlmao.png", img)