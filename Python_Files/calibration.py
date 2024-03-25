import sys
import cv2 as cv
import numpy as np
import time
# import imutils

#### Camera parameters to undistort and rectify images

cv_file = cv.FileStorage()
cv_file.open('stereoMap.xml', cv.FileStorage_READ)

stereoMap0_x = cv_file.getNode('stereoMap0_x').mat()
stereoMap0_y = cv_file.getNode('stereoMap0_y').mat()
stereoMap1_x = cv_file.getNode('stereoMap1_x').mat()
stereoMap1_y = cv_file.getNode('stereoMap1_y').mat()


def undistortRectify(image0, image1):
    
    ##undistort and rectify images
    
    undistorted0 = cv.remap(image0, stereoMap0_x, stereoMap0_y, cv.INTER_LANCZOS4)
    undistorted1 = cv.remap(image1, stereoMap1_x, stereoMap1_y, cv.INTER_LANCZOS4)
    
    return undistorted0, undistorted1