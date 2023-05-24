import cv2
import numpy as np

img = cv2.imread('/Users/kouki/Image_practice/sample/frame/sample_video1_gray_0100.jpg')

while True:
    cv2.imshow("scale",img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyWindow()