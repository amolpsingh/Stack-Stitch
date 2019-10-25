import numpy as np
import cv2
#help(cv2.xfeatures2d)
#from matplotlib import pyplot as plt

#%% defines relationship of images to be stacked
def findHomography(image_1_kp, image_2_kp, matches):
    #makes two empty arrays for keypoints of images
    img1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    img2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)

    #adds key points of each image to arrays
    for i in range(0,len(matches)):
        img1_points[i] = image_1_kp[matches[i].queryIdx].pt
        img2_points[i] = image_2_kp[matches[i].trainIdx].pt

    #creates homography for entire input images using the key points
    #Uses RANSAC algorithm to find best feature matching points and their corresponding (x,y) coordinate
    homography, mask = cv2.findHomography(img1_points, img2_points, cv2.RANSAC, ransacReprojThreshold=2.0)

    return homography
#%%
# function to align the images
def imageAlignment(images):

    ret = []
    
    #use SIFT through sub-module xfeatures2d 
    sift = cv2.xfeatures2d.SIFT_create()
    
    #image 0 is base image and align everything to it
    ret.append(images[0])
    baseImage = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    
    #finds keypoints and descriptors in single step
    kp_base, des_base = sift.detectAndCompute(baseImage, None)

    #for image to draw keypoints 
    #img = cv2.drawKeypoints(baseImage,kp_base,images[0])
    #cv2.imwrite('sift_keypoints.jpg',img)

    #loop through the list of images
    for img in range(1, len(images)):
        kp_img, des_img = sift.detectAndCompute(images[img], None)
        #brute-force matcher that takes descriptor of one feature
        bf = cv2.BFMatcher(cv2.NORM_L2)
        #returns 2 best matches for all descriptor
        matches = bf.knnMatch(des_img, des_base, k=2)
        
        # Apply ratio test from https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        strongMatches = []
        for m,n in matches:
            if m.distance < 0.80*n.distance:
                strongMatches.append(m)
        
        #order the matches from nearest to farthest
        sortMatches = sorted(strongMatches, key=lambda x: x.distance, reverse = False)
        #select only first 128 labeled points https://programtalk.com/python-examples-amp/cv2.SIFT/
        #gradient orientation histograms of several small windows (128 values for each point)
        matches=sortMatches[0:128]
                
        #the homography tells you how to warp the images so that they're in the
        #same coordinate frame to be placed on top of each other for stacking
        M = findHomography(kp_img, kp_base, matches )
        
        #find size of initial image
        size = (images[img].shape[1], images[img].shape[0])
        
        #application of tranformation of images[img] using matrix M, obtained 
        #from findHomography, and final params indicate width and height of 
        #final output image to apply transformation so images can be stacked
        newImage = cv2.warpPerspective(images[img], M, dsize = size, flags=cv2.INTER_LINEAR)
        
        #adds images to list after alignment
        ret.append(newImage)
        
    return ret

#%%
    # draw keypoint map
    
#img1 = cv2.imread("/Users/amolsingh/Documents/OhgamiLab/Images/Stacking Images/stacking/Q1.png")
#img2 = cv2.imread("/Users/amolsingh/Documents/OhgamiLab/Images/Stacking Images/stacking/Q1a.png") 
#
##use SIFT through sub-module xfeatures2d 
#sift = cv2.xfeatures2d.SIFT_create()
#
##finds keypoints and descriptors in single step
#kp1, des1 = sift.detectAndCompute(img1, None)
##tests
##kp2, des2 = sift.detectAndCompute(img2, None)
#
#img = cv2.drawKeypoints(img1, kp1, img1)
#cv2.imwrite('sift_keypoints.jpg',img)
#bf = cv2.BFMatcher(cv2.NORM_L2)
#returns 2 best matches for all descriptor
#matches = bf.knnMatch(des1, des2, k=2)
    
# Apply ratio test from https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
#strongMatches = []
#for m,n in matches:
#    if m.distance < 0.75*n.distance:
#        strongMatches.append([m])
#
## draw the matched btw images
#img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, strongMatches, None, singlePointColor = (0,0,0), matchColor = (172,142,142), flags = 0)
#cv2.imwrite("/Users/amolsingh/Documents/OhgamiLab/StackingPaper/Images/sift keypoint map.png", img3)
#plt.imshow(img3), plt.show()
        
      

