import numpy as np
import cv2 as cv
import matplotlib.pyplot
import glob
import time

class worldCOS:
    
    rotVec = None
    transVec = None
    camMatrix = None
    disicoeff = None
    
    def __init__(self):
        self.rotVec = np.load('pixel2WCos/rot_mtx.npy') 
        self.transVec = np.load('pixel2WCos/tTvec_mtx.npy')
        self.camMatrix = np.load('pixel2WCos/cam_mtx.npy')
        self.disicoeff = np.load('pixel2WCos/dist.npy')
        
        
    def calculateXYZ(self,u,v):
                
        pixel = np.array([[u,v,1]], dtype= np.float32)
        pixel = pixel.T
        invCamMat = np.linalg.inv(self.camMatrix)
        
        rotMat,_ = cv.Rodrigues(self.rotVec)
        
        invRotMat = np.linalg.inv(rotMat)
        
        xyz = invRotMat.dot(invCamMat.dot(pixel) - self.transVec)
        
        
        return xyz
    
    
    

def caliberation(image):
    
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    img = cv.imread(image)
    
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
    ret, camMat, dist, rotVecs, transVecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # np.save('pixel2WCos/cam_mtx.npy', camMat)
    # np.save('pixel2WCos/dist.npy', dist)
    # np.save('pixel2WCos/rot_mtx.npy', rotVecs)
    # np.save('pixel2WCos/tTvec_mtx.npy', transVecs)
    print("camera_Matrix: \n", camMat)
    print("Distortion coefficient: \n", dist)
    print("Rotation Vector: \n", rotVecs)
    print("Trnaslation Vector: \n", transVecs)
    
# img = "Images\cam1Img\image1_2.bmp" 
# caliberation(img)


    
    
def pixel2World(pixelValues, camMatrix, distCoeff, rotVector, transVector):
    
    # Convert Pixel coordinates to normalized image coordinates
    # x = (u-cx)/fx; y = (v-cy)/fy
    normalCoord = cv.undistortPoints(np.array([pixelValues],dtype = np.float32), camMatrix, distCoeff)
    print(camMatrix)
    print(pixelValues )
    print(normalCoord)
    
    # Convert Normalized coordinates to homogenous coordinates
    homogenousCoord = np.append(normalCoord[0][0], 1)
    homogenousCoord = np.array([homogenousCoord], dtype= np.float32)
    homogenousCoord = homogenousCoord.T    
    
    # Formaula : 
    
    # pixelValues = Intrinisic Matrix * image_coordinates
    # image_coordinates =  Roatation_Matrix + World_COS * Trasnlation Vector
    # pixelValues = Intrinisic Matrix *( Roatation_Matrix + World_COS * Trasnlation Vector)
    # World_COS = inv(Rotation_Matrix)*((inv(Intrinsic_Matrix)* pixelValues) - Translation_Vector)
    
    rotMatrix, s = cv.Rodrigues(rotVector)
    invRotMatrix = np.linalg.inv(rotMatrix)
    invCamMatrix = np.linalg.inv(camMatrix)
    worldCos1 = np.dot(invRotMatrix, (homogenousCoord - transVector))
    print(worldCos1)
    
    ## 2nd method
    pixel = np.array([[pixelValues[0], pixelValues[1], 1]], dtype=np.float32)
    # pixel = np.array([[homogenousCoord[0], homogenousCoord[1], 1]], dtype=np.float32)
    pixel = pixel.T
    
    
    worldCOS = np.dot(invRotMatrix, (np.dot(invCamMatrix,pixel)- transVector))  
    
    return worldCOS

rotVec = np.load('pixel2WCos/rot_mtx.npy') 
transVec = np.load('pixel2WCos/tTvec_mtx.npy') 
camMatrix = np.load('pixel2WCos/cam_mtx.npy')
disicoeff = np.load('pixel2WCos/dist.npy')
pixel = (200,200)


path = "Images\cam1Img\image1_2.bmp" 
img = cv.imread(path)
print(img.shape)

x= img[pixel]
cv.circle(img, (x[0],x[1]), 20,(0, 0,255), -1)
cv.imshow("Edt image",img)
cv.waitKey(0)
cv.destroyAllWindows()



wcs = pixel2World(pixel, camMatrix, disicoeff, rotVec, transVec)  
print(wcs)

c1 = worldCOS()
print(c1.calculateXYZ(200,200))
    


