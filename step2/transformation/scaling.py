import cv2
import numpy as np

img = cv2.imread('/Users/kouki/Image_practice/sample/frame/sample_video1_gray_0100.jpg')
res = cv2.resize(img,None,fx=14,fy=5,interpolation=cv2.INTER_CUBIC)

cv2.imshow("Resize",res)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('../../sample/frame/scaling.png',res)
    cv2.destroyAllWindows()