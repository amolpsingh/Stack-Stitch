#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:05:29 2018

@author: amolsingh
"""

import cv2
import VideoPrep
import numpy as np
import time, os
import matplotlib.pyplot as plt

#%% Main
def stitch(folder):
    time1 = time.time()
    VideoPrep.vid2imgs(folder)
    dir_path = os.path.dirname(folder)
    newpath = os.path.join(dir_path, "QuadOutputFrames")
    imgs = VideoPrep.findFrames(newpath)
#    intimgs = VideoPrep.chooseFrame(newpath)
#    imgs=[]
#    print(str(len(intimgs)))
#    for i in range (0, len(intimgs)):
#        paus = intimgs[i]
#        img = np.array(paus)
#        imgs.append(img)
#    for i in range(0, len(imgs)):
#        print("count: " + str(i))
#        plt.imshow(imgs[i])
#        plt.show() 
    stitcher = cv2.createStitcher(False) 
    result = np.empty(shape=[2048, 2048])
    ret, result = stitcher.stitch(imgs, result)
#    cv2.imwrite("/Users/amolsingh/Documents/OhgamiLab/Videos/QuadOutputFrames/QuadStitched.jpg", result)
    cv2.imwrite(os.path.join(newpath, "QuadStitched.jpg"), result)
    time2 = time.time()
    fintime = time2 - time1
    print("Time: " + str(fintime))

#%%
#stitch("/Users/amolsingh/Documents/OhgamiLab/Videos/slovid27-12.MOV")
#stitch("/Users/amolsingh/Documents/OhgamiLab/Videos/normvid17-12.MOV")






