import sys
import numpy as np
import cv2 as cv
import time
import glob
import matplotlib.pyplot as plt


# Load the camera parameters to undistort and rectify images

cv_file = cv.FileStorage()
cv_file.open('pixel2WCos/stereoMap.xml', cv.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
camMatrixR = np.load('pixel2WCos/calibParams/cameraMatrixRight.npy')
camMatrixL = np.load('pixel2WCos/calibParams/cameraMatrixLeft.npy')
distcoefR = np.load('pixel2WCos/calibParams/Distortion_CoefficentRight.npy')
distcoefL = np.load('pixel2WCos/calibParams/Distortion_CoefficentLeft.npy')
newCamMatR = np.load('pixel2WCos/calibParams/newcameraMatrixRight.npy')
newCamMatL = np.load('pixel2WCos/calibParams/newcameraMatrixLeft.npy')
transVect = np.load('pixel2WCos/calibParams/TranslationVectorLR.npy')
rectTransL = np.load('pixel2WCos/calibParams/rotationMatrixLeft.npy')
rectTransR = np.load('pixel2WCos/calibParams/rotationMatrixRight.npy')
projMatrixL = np.load('pixel2WCos/calibParams/projectionMatrixLeft.npy')
projMatrixR = np.load('pixel2WCos/calibParams/projectionMatrixRight.npy')


baseline = np.linalg.norm(transVect)
focalLength = camMatrixL[0,0]

def UndistortImage( imageL, imageR):
    
    undistortL = cv.remap(imageR, stereoMapL_x, stereoMapL_y, cv.INTER_LINEAR )
    undistortR = cv.remap(imageL, stereoMapR_x, stereoMapR_y, cv.INTER_LINEAR )
    

    return undistortL, undistortR


def undistortImg(imageL, imageR):

    undistL = cv.undistort(imageL, camMatrixL, distcoefL, None, newCamMatL)
    undistR = cv.undistort(imageR, camMatrixR, distcoefR, None, newCamMatR)
    
    return undistL, undistR

def disparityMap(imgL, imgR):
    
    stereo = cv.StereoBM.create(numDisparities= 16, blockSize= 15)
    disparity = stereo.compute(imgL, imgR)
    
    return disparity



images0 = glob.glob("pixel2WCos/TestImages/Cam0/*.bmp")
images1 = glob.glob("pixel2WCos/TestImages/Cam1/*.bmp")

for imageR, imageL in zip(images0, images1):
    
    imageL = cv.imread(imageL,0)
    imageR = cv.imread(imageR,0)
    
    undistLeft, undistRight = UndistortImage(imageL, imageR)
    
    undistL, undistR = undistortImg(imageL, imageR)
    cv.imshow("remap_Left", undistLeft)
    cv.imshow("remap_Right", undistRight)
    
    
    # cv.imshow("undistortLeft", undistL)
    # cv.imshow("undistortRight", undistR)    
    
    # disparity_map = disparityMap(undistLeft, undistRight)   
    # # plt.imshow( disparity, 'gray')
    # # plt.show()
    
    # depth_map = (baseline * focalLength) / (disparity_map + 1e-6)  
    
    # depth_map_visual = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    
    # # Normalize depth map for visualization
    # depth_map_normalized = cv.normalize(depth_map, None, 0, 1, cv.NORM_MINMAX)
    # color_map = cv.applyColorMap((depth_map_normalized * 255).astype(np.uint8), cv.COLORMAP_JET)
    # cv. imshow('Depth Map', color_map)
    cv.waitKey(0)
   
plt.show()
    
cv.destroyAllWindows()

print(baseline)
print(focalLength)
