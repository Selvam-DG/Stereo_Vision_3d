
# ############### Get Intrinscic and Extrinsic Parametes by CALIBRATION ########################
# calibration, cameraMatrix, distotion, rotVector, transVector = cv.calibrateCamera(objPoints, imgPoints, framesize, None, None)

# print("Camera Calibrated \n", calibration)
# print("Distortions =  \n", distotion)
# print("Roatation Vectot = \n", rotVector)
# print("Translation Vector \n", transVector)



# ################# UNDISTORTION ###########################
# img = cv.imread('image0_7.png')
# height, width = img.shape[:2]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distotion, (width,height), 1, (width,height))


# ########## undistort with undistort function
# dst = cv.undistort(img, cameraMatrix, distotion, None, newcameramtx)
 
# # crop the image
# x, y, width, height = roi
# dst = dst[y:y+height, x:x+width]
# cv.imwrite('calibresult.png', dst)

# ##### undistort with remapping
# mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, distotion, None, newcameramtx, (width,height), 5)
# dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
 
# # crop the image
# x, y, width, height = roi
# dst = dst[y:y+height, x:x+width]
# cv.imwrite('calibresult.png', dst)

# ########### REPROJECTION ERROR ###########

# mean_error = 0
# for i in range(len(objPoints)):
#  imgpoints2, _ = cv.projectPoints(objPoints[i], rotVector[i], transVector[i], cameraMatrix, distotion)
#  error = cv.norm(imgPoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
#  mean_error += error
 
# print( "\n total error: {}".format(mean_error/len(objPoints)) )

