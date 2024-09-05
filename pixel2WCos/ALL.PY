import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt

def undistortImg(image0, image1):
    camMatrix0 = np.load('pixel2WCos/calibParams/cameraMatrix0.npy')
    camMatrix1 = np.load('pixel2WCos/calibParams/cameraMatrix1.npy')
    distcoef0 = np.load('pixel2WCos/calibParams/Distortion_Coefficent0.npy')
    distcoef1 = np.load('pixel2WCos/calibParams/Distortion_Coefficent1.npy')
    newCamMat0 = np.load('pixel2WCos/calibParams/newcameraMatrix0.npy')
    newCamMat1 = np.load('pixel2WCos/calibParams/newcameraMatrix1.npy')
    
    #The function is simply a combination of initUndistortRectifyMap (with unity R ) and remap (with bilinear interpolation)
    undistorted0 = cv.undistort(image0, camMatrix0, distcoef0, newCamMat0)
    undistorted1 = cv.undistort(image1, camMatrix1, distcoef1, newCamMat1)
    
    return undistorted0, undistorted1

def undistortRectify(image0, image1):
    
    cv_file = cv.FileStorage()
    cv_file.open('pixel2WCos/stereoMap.xml', cv.FileStorage_READ)
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
    stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
    stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
    ##undistort and rectify images
    
    undistortedR = cv.remap(image0, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4)
    undistortedL = cv.remap(image1, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4)
 
    return undistortedR, undistortedL

images0 = glob.glob("pixel2WCos/TestImages/Cam0/*.bmp")
images1 = glob.glob("pixel2WCos/TestImages/Cam1/*.bmp")

for imageR, imageL in zip(images0, images1):
    imageR = cv.imread(imageR,0)
    imageL = cv.imread(imageL,0)
    
    # undist0, undist1 = undistortImg(imageR, imageL)
    undist0, undist1 = undistortRectify(imageR, imageL)
    cv.imshow("undistort1", undist0)
    cv.imshow("undistortL", undist1)
    
    
    stereo = cv.StereoBM.create(numDisparities= 16, blockSize= 15)
    disparity = stereo.compute(undist0, undist1)
    # plt.imshow( disparity, 'gray')
    # plt.show()
    # # cv.imshow("disparity",disparity)
    cv.waitKey(0)
cv.destroyAllWindows()
