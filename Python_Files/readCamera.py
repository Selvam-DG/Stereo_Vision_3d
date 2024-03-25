import cv2 as cv
import numpy as np

# cap0 = cv.VideoCapture(0)
cap1 = cv.VideoCapture(0)

count = 0

while cap1.isOpened():
    
    # ret0, img0 = cap0.read()
    ret1, img1 = cap1.read()
    
    k = cv.waitKey(1) & 0xFF
    
    if k == 27:
        break
    elif k == ord('s'):
        # cv.imwrite('Images/cam0Img/image0_' + str(count) + '.png', img0)
        cv.imwrite('Images/cam1Img/image1_' + str(count)+ '.png', img1)
        print("Images Saved")
        count+=1
        cv.destroyAllWindows()
        
    # cv.imshow("Cam0_image", img0)
    cv.imshow("Cam1_image", img1)
        
# cap0.release()
cap1.release()
