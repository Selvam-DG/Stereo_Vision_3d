import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt

def StereoCaliberation():
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
    imgPointsL = [] # 2d points in image plane of camera1.
    imgPointsR = [] # 2d points in image plane of camera0.

    imagesL = glob.glob("Images/camL/*.bmp")
    imagesR = glob.glob("Images/camR/*.bmp")


    for imageL, imageR in zip(imagesL, imagesR):
        
        imgL = cv.imread(imageL)
        imgR = cv.imread(imageR)

        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
        
        
        #Find corners in the chessboard
        retL, cornersL = cv.findChessboardCorners(grayL, boardsize, None)
        retR, cornersR = cv.findChessboardCorners(grayR, boardsize, None)
        
        
        # If corners are found, add object points and image points
        if retL and retR == True:
            objPoints.append(objP)
            
            #append the chessboard corner point to imagepoints
            cornersL = cv.cornerSubPix(grayL, cornersL, (11,11),(-1,-1), criteria)
            imgPointsL.append(cornersL)
            cornersR = cv.cornerSubPix(grayR, cornersR, (11,11),(-1,-1), criteria)
            imgPointsR.append(cornersR)

            
            # draw and display the corners
            cv.drawChessboardCorners(imgL, boardsize,cornersL, retL)
            cv.imshow("camera1_img", imgL)        
            cv.drawChessboardCorners(imgR, boardsize,cornersR, retR)
            cv.imshow("camera0_img", imgR)

            
            cv.waitKey(1000)
            
    cv.destroyAllWindows()
            

    ##############   CALIBRATION ##############################################
    #Camera1 Caliberation
    calibration1L, cameraMatrixL, distortionL, rotVectorL, transVectorL = cv.calibrateCamera(objPoints, imgPointsL, grayL.shape[::-1], None, None)
    heightL, widthL, channelsL = imgL.shape
    #  # #### Returns the new camera intrinsic matrix based on the free scaling parameter.
    newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distortionL,(widthL, heightL), 1, (widthL, heightL))

    # Camera2 Caliberation
    calibrationR, cameraMatrixR, distortionR, rotVectorR, transVectorR = cv.calibrateCamera(objPoints, imgPointsR, grayR.shape[::-1], None, None)
    heightR, widthR, channelsR = imgR.shape
    #  # #### Returns the new camera intrinsic matrix based on the free scaling parameter.
    newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distortionR,(widthR, heightR), 1, (widthR, heightR))
    
    ################ Stereo Vision Caliberation #######################
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    # here we fix the intrinsic camera matrices so that only Rotation, tanslataion essential and fundamental matrices
    # so the intrinsic parameters are the same

    ret, newCameraMatrixL, distCoeffsL, newCameraMatrixR, distCoeffsR, rotMatrix, transVector, essMatrix, fundMatrix = cv.stereoCalibrate(objPoints,  imgPointsL, imgPointsR, newCameraMatrixL, distortionL,
                                                                                                                            newCameraMatrixR, distortionR, grayR.shape[::-1], criteria=criteria, flags=flags)



    ############# Stereo Rectification #######################################
    # #  Computes rectification transforms for each head of a calibrated stereo camera
    # # # alpha = 1
    # # #alpha=0 means that the rectified images are zoomed and shifted   
    rectifyScale = 1
    rectTransL, rectTransR, projMatrixL, projMatrixR, disp2DepthMapMatrix, roiL, roiR = cv.stereoRectify(newCameraMatrixL, distCoeffsL, newCameraMatrixR, distCoeffsR, grayR.shape[::-1], rotMatrix, transVector, rectifyScale, (0,0))


    stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distCoeffsL, rectTransL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)

    stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distCoeffsR, rectTransR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

    # print('stereoMapR_x', stereoMapR[0])
    # print('stereoMapR_y', stereoMapR[1])
    # print('stereoMapL_x', stereoMapL[0])
    # print('stereoMapL_y', stereoMapL[1])
    # print("Saving Caliberation Parameters")
    
    # np.save('pixel2WCos/calibParams/cameraMatrixLeft.npy', cameraMatrixL)
    # np.save('pixel2WCos/calibParams/cameraMatrixRight.npy', cameraMatrixR)

    # np.save('pixel2WCos/calibParams/newcameraMatrixLeft.npy', newCameraMatrixL)
    # np.save('pixel2WCos/calibParams/newcameraMatrixRight.npy', newCameraMatrixR)

    # np.save('pixel2WCos/calibParams/Distortion_CoefficentLrft.npy', distCoeffsL)
    # np.save('pixel2WCos/calibParams/Distortion_CoefficentRight.npy', distCoeffsR)

    # np.save('pixel2WCos/calibParams/RotationMatrixLR.npy', rotMatrix)
    # np.save('pixel2WCos/calibParams/TranslationVectorLR.npy', transVector)
    # np.save('pixel2WCos/calibParams/projectionMatrixLeft.npy', projMatrixL)
    # np.save('pixel2WCos/calibParams/projectionMatrixRight.npy', projMatrixR)

    # np.save('pixel2WCos/calibParams/rotationMatrixLeft.npy', rectTransL)
    # np.save('pixel2WCos/calibParams/rotationMatrixRight.npy', rectTransR)

    # np.save('pixel2WCos/calibParams/disparity2depthMapMatrix.npy',disp2DepthMapMatrix)



    print("saving Parameters")
    cv_file = cv.FileStorage('pixel2WCos/stereoMap.xml', cv.FILE_STORAGE_WRITE)

    cv_file.write('stereoMapL_x', stereoMapL[0])
    cv_file.write('stereoMapL_y', stereoMapL[1])
    cv_file.write('stereoMapR_x', stereoMapR[0])
    cv_file.write('stereoMapR_y', stereoMapR[1])

    cv_file.release()



StereoCaliberation()





# undist0, undist1 = undistort(img0, img1)   
 
# plt.subplot(2,2,1)
# plt.imshow( img0, 'gray')
# plt.subplot(2,2,2)
# plt.imshow( img1, 'gray')
# plt.subplot(2,2,3)
# plt.imshow( undist0, 'gray')
# plt.subplot(2,2,4)
# plt.imshow( undist1, 'gray')
# plt.show()




# stereo = cv.StereoBM.create(numDisparities= 16, blockSize= 3)
# disparity = stereo.compute(undist0, undist1)
# plt.subplot(2,2,1)
# plt.imshow(img0, 'gray')
# plt.subplot(2,2,2) 
# plt.imshow(img1, 'gray')
# plt.subplot(2,2,3)
# plt.imshow( disparity, 'gray')
# plt.show()

# Q = np.load('pixel2WCos/calibParams/disparity2depthMapMatrix.npy') 


# Image3D = cv.reprojectImageTo3D(disparity= disparity, Q= Q ) 
# cv.imshow("3D Image", Image3D)
# cv.waitKey(0)
# cv.destroyAllWindows()


# # ################## TRIANGULATE POINTS ######################################

# projMatrix0 = np.load('pixel2WCos/calibParams/projectionMatrix0.npy')
# projMatrix1 = np.load('pixel2WCos/calibParams/projectionMatrix1.npy')
# u,v = 200,200
# print("\nP0: \n", projMatrix0)
# print("\nP1: \n", projMatrix1)
# disparityPix = disparity[v, u]
# focLen = projMatrix0[0,0]
# print(projMatrix1[0,3])
# baseline = -projMatrix1[0,3] / projMatrix1[0,0]
# depth = focLen *baseline / ( disparity + 1e-6)
# print("Depth : ", depth)
# print("Baseline: ", baseline)

# projPoints0 = np.array([[u, v]], dtype = np.float32 ).reshape(-1, 1, 2)
# projPoints1 = np.array([[u, v]], dtype = np.float32 ).reshape(-1, 1, 2)
# # projPoints1 = np.array([[[i, j]] for j in range(remap1.shape[0]) for i in range(remap1.shape[1])]).reshape(-1, 1, 2).astype(np.float32)

# points4D = cv.triangulatePoints(projMatrix0, projMatrix1, projPoints0, projPoints1 )

# point3d = points4D[:3] / points4D[3]
# print(points4D)
# print("\n \n 3d Points\n" ,point3d)

# print(Q)