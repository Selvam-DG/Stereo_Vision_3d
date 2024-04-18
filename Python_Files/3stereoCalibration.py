import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt

######### FIND CHESSBOARD CORNERS #######################
boardsize = (9,7)
framesize = (640,480)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#prepare object points with numpy
objP = np.zeros((boardsize[0]*boardsize[1],3), np.float32)
objP[:,:2] = np.mgrid[0:boardsize[0],0:boardsize[1]].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objPoints = [] # 3d point in real world space
imgPoints0 = [] # 2d points in image plane of camera0.
imgPoints1 = [] # 2d points in image plane of camera1.

images0 = glob.glob("Images/cam0Img/*.bmp")
images1 = glob.glob("Images/cam1Img/*.bmp")

for image0, image1 in zip(images0, images1):
    img0 = cv.imread(image0)
    img1 = cv.imread(image1)
    
    gray0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    
    #Find corners in the chessboard
    ret0, corners0 = cv.findChessboardCorners(gray0, boardsize, None)
    ret1, corners1 = cv.findChessboardCorners(gray1, boardsize, None)
    
    # If corners are found, add object points and image points
    if ret0 and ret1 == True:
        objPoints.append(objP)
        
        #append the chessboard corner point to imagepoints
        corners0 = cv.cornerSubPix(gray0, corners0, (11,11),(-1,-1), criteria)
        imgPoints0.append(corners0)
        corners1 = cv.cornerSubPix(gray1, corners1, (11,11),(-1,-1), criteria)
        imgPoints1.append(corners1)
        
        # draw and display the corners
        
        cv.drawChessboardCorners(img0, boardsize,corners0, ret0)
        cv.imshow("camera0_img", img0)
        cv.drawChessboardCorners(img1, boardsize,corners1, ret1)
        cv.imshow("camera1_img", img1)
        
        cv.waitKey(1000)
        
cv.destroyAllWindows()
        

##############   CALIBRATION ##############################################

calibration0, cameraMatrix0, distotion0, rotVector0, transVector0 = cv.calibrateCamera(objPoints, imgPoints0, gray0.shape[::-1], None, None)
height0, width0, channels0 = img0.shape
newCameraMatrix0, roi_0 = cv.getOptimalNewCameraMatrix(cameraMatrix0, distotion0,(width0, height0), 1, (width0, height0))



calibration1, cameraMatrix1, distotion1, rotVector1, transVector1 = cv.calibrateCamera(objPoints, imgPoints1, gray1.shape[::-1], None, None)
height1, width1, channels1 = img1.shape
newCameraMatrix1, roi_1 = cv.getOptimalNewCameraMatrix(cameraMatrix1, distotion1,(width1, height1), 1, (width1, height1))


################ Stereo Vision Caliberation #######################
flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
# here we fix the intrinsic camera matrices so that only Rotation, tanslataion essential and fundamental matrices
# so the intrinsic parameters are the same

ret, newCameraMatrix0, distCoeffs0, newCameraMatrix1, distCoeffs1, rotVector, transVector, essMatrix, fundMatrix = cv.stereoCalibrate(objPoints,  imgPoints0, imgPoints1, newCameraMatrix0, distotion0,
                                                                                                                           newCameraMatrix1, distotion1, gray0.shape[::-1])

print(newCameraMatrix0)
print(newCameraMatrix1)
print('*************'*20)


############# Stereo Rectification #######################################
    
rectifyScale = 1
rectin0, rectin1, projMatrix0, projMatrix1, ddMapMatrix, roi0, roi1 = cv.stereoRectify(newCameraMatrix0, distCoeffs0, newCameraMatrix1, distCoeffs1, gray0.shape[::-1], rotVector, transVector, rectifyScale, (0,0))


stereoMap0 = cv.initUndistortRectifyMap(newCameraMatrix0, distCoeffs0, rectin0, projMatrix0, gray0.shape[::-1], cv.CV_16SC2)

stereoMap1 = cv.initUndistortRectifyMap(newCameraMatrix1, distCoeffs1, rectin1, projMatrix1, gray1.shape[::-1], cv.CV_16SC2)

print(stereoMap0)
print("##############"*10)
print(stereoMap1)

print('stereoMap0_x', stereoMap0[0])
print('stereoMap0_y', stereoMap0[1])
print('stereoMap1_x', stereoMap1[0])
print('stereoMap0_y', stereoMap0[1])



# print("saving Parameters")
# cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

# cv_file.write('stereoMap0_x', stereoMap0[0])
# cv_file.write('stereoMap0_y', stereoMap0[1])
# cv_file.write('stereoMap1_x', stereoMap1[0])
# cv_file.write('stereoMap1_y', stereoMap1[1])

# cv_file.release()

