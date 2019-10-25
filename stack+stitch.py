#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 16:28:35 2018

@author: amolsingh
"""
import VideoPrep
import StackingMain, runner
import cv2
import os
import time
import numpy as np 

def stackNstitch(folder1, folder2):  
#    time1 = time.time()
#    VideoPrep.vid2imgs(folder1, 1)
    dir_path1 = os.path.dirname(folder1)
    newpath1 = os.path.join(dir_path1, "SSOutputFrames1")
    imgs1 = VideoPrep.findFrames(newpath1)
#    VideoPrep.vid2imgs(folder2, 2)
#    dir_path2 = os.path.dirname(folder2)
#    newpath2 = os.path.join(dir_path2, "QuadOutputFrames2")
#    imgs2 = VideoPrep.findFrames(newpath2)
    
    stitcher = cv2.createStitcher(False) 
    result = np.empty(shape=[2048, 2048])
    ret, result = stitcher.stitch(imgs1, result)
    cv2.imwrite(os.path.join(newpath1, "QuadStitched1.jpg"), result)
#    time2 = time.time()
#    fintime = time2 - time1
#    print("Time: " + str(fintime))

#stackNstitch("/Users/amolsingh/Documents/OhgamiLab/Videos/stackstitchtry1", "/Users/amolsingh/Downloads/cropped2.mov" )
#runner.stitch("/Users/amolsingh/Downloads/QuadOutputFrames1")

#%%
def stackNstitch1vid(folder):
    time1=time.time()
    VideoPrep.vid2imgs1vid(folder)
    for n in range(1, 486):
        stacked = StackingMain.stacker("/Users/amolsingh/Documents/OhgamiLab/Videos/FramesStackNStitch/OutputFrames%d" %n)
        cv2.imwrite("/Users/amolsingh/Documents/OhgamiLab/Videos/FramesStackNStitch/final/frame%d.jpg" %n, stacked )
    
    path, dirs, files = os.walk("/Users/amolsingh/Documents/OhgamiLab/Videos/FramesStackNStitch/final").__next__()
#    path, dirs, files = os.walk("/Users/amolsingh/Documents/OhgamiLab/Videos/FramesStackNStitch/OutputFrames(singlebatch)").__next__()
    size = len(files)
    inp = []
    for i in range(1, size):
        a = cv2.imread("/Users/amolsingh/Documents/OhgamiLab/Videos/FramesStackNStitch/final/frame%d.jpg" %i)
#        a = cv2.imread("/Users/amolsingh/Documents/OhgamiLab/Videos/FramesStackNStitch/OutputFrames(singlebatch)/frame%d.jpg" %i)
        inp.append(a)
        
    result = np.empty(shape=[4000, 4000])
    stitcher = cv2.createStitcher(False) 
    ret, result = stitcher.stitch(inp, result)
    cv2.imwrite("/Users/amolsingh/Documents/OhgamiLab/Videos/FramesStackNStitch/final/stitched.jpg", result)
#    cv2.imwrite("/Users/amolsingh/Documents/OhgamiLab/Videos/FramesStackNStitch/final/stitched(singlebatch).jpg", result)
    time2 = time.time( )
    fintime = time2 - time1
    print("Time: " + str(fintime))

stackNstitch1vid("/Users/amolsingh/Documents/OhgamiLab/Videos/stackstitchtry1.mov")