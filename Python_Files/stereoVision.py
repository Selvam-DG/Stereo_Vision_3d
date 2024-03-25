import sys
import cv2 as cv
import numpy as np
import time
##### import imutils
import matplotlib.pyplot as plt

##### Functions for stereo vision and depth estimation
import depthCal as depth
import calibration

##### Media pipe for binpicking part

##### open camera
cam0 = cv.VideoCapture(0, cv.CAP_DSHOW)
cam1 = cv.VideoCapture(1, cv.CAP_DSHOW)


#### setup Parameter
# frame_rate = 















