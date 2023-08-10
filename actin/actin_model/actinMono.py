import cv2
import numpy as np

img=cv2.imread("../actin_image/sampleMono.png")

white_mask=cv2.inRange(img,np.array([200,200,200]),np.array([255,255,255]))
img[white_mask == 255] = [0,0,0]

cv2.imshow("white_png",white_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
