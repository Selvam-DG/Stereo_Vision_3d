import numpy as np
import cv2 as cv
import matplotlib

class stereovision:
    
    rotVec = None
    transVec = None
    camMatrix = None
    distcoeff = None
    
    def viewImage(self, image):
        
        cv.imshow("Image",image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    def caliberate(self, image):

    
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((9*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        
        img = image
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,9), None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2=cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (7,9), corners2, ret)
            cv.imshow("DrawChessBoard", img)
            cv.waitKey(1000)

        cv.destroyAllWindows()
        ret, self.camMatrix, self.distcoeff, self.rotVec, self.transVec = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        # np.save('pixel2WCos/cam_mtx.npy', self.camMat)
        # np.save('pixel2WCos/dist.npy', self.dist)
        # np.save('pixel2WCos/rot_mtx.npy', self.rotVecs)
        # np.save('pixel2WCos/tTvec_mtx.npy', self.transVecs)
        print("camera_Matrix: \n", self.camMatrix)
        print("Distortion coefficient: \n", self.distcoeff)
        print("Rotation Vector: \n", self.rotVec)
        print("Trnaslation Vector: \n", self.transVec)
    
    
    def calculateWorldCOS(self, u,v):
        
   
        pixel = np.array([[u,v,1]], dtype= np.float32)
        pixel = pixel.T
        invCamMat = np.linalg.inv(self.camMatrix)
        rotMat,_ = cv.Rodrigues(self.rotVec[0])
        invRotMat = np.linalg.inv(rotMat) 
        xyz = invRotMat.dot(invCamMat.dot(pixel) - self.transVec[0])        
        
        return xyz

s1 = stereovision()

image = cv.imread("Images\cam1Img\image1_2.bmp" )
s1.viewImage(image)
# s1.caliberate(image)
print(s1.calculateWorldCOS(200,200))