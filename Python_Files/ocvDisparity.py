import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
imgL = cv.imread('Images\cam0Img\image1.bmp', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('Images\cam1Img\image1_1.bmp', cv.IMREAD_GRAYSCALE)
# cv.imshow("Image", imgL)
# cv.waitKey(0)

stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()
cv.destroyAllWindows()