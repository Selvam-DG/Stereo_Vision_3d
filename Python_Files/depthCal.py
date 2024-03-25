import sys
import cv2 as cv
import numpy as np
import time

def  find_depth(img0CenterPoint, img1CentrePoint, image0, image1, baseline, focalLen, alpha ):
    
    ## convert Focal length(mm) to pixels
    height0, width0, depth0 = image0.shape
    height1, width1, depth1 = image1.shape
    
    if width0 == width1:
        focal_pixel = (width0 *0.5) / np.tan(alpha * 0.5 *np.pi/180)
    else:
        print('Camera0 and Camera1  frames do not have the same pixel width')
    
    x0 = img0CenterPoint[0]
    x1 = img1CentrePoint[0]
    
    ## calculate the disparity that is displacement between camera0 and camera1
    disparity = x0 - x1

    ## calculate the depth z:
    depth = (baseline * focal_pixel) / disparity
    
    return abs(depth)
    