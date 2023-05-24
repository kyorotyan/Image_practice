import cv2
import numpy as np

img = cv2.imread('/Users/kouki/Image_practice/sample/frame/sample_video1_gray_0100.jpg')
res = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)