import cv2
import numpy as np

img = cv2.imread("/Users/kouki/Image_practice/sample/frame/sample_video1_gray_0100.jpg")

#画像の中心を計算
height,width=img.shape[:2]

#画像のリサイズ(1/2に)
center=(width//2,height//2)

#回転角度とスケールを固定
angle=35 #回転角度(度数)
scale=1.0 #スケール(1.0で等倍)

# 回転行列を計算
rotation_matrix=cv2.getRotationMatrix2D(center,angle,scale)
#画像の回転
rotated_image=cv2.warpAffine(img,rotation_matrix,(width,height))

cv2.imshow("Rotated.img",rotated_image)

k=cv2.waitKey(0)
if k==ord("q"):
    cv2.destroyAllWindows()
elif k==ord("s"):
    cv2.imwrite('rotated.png',rotated_image)
    cv2.destroyAllWindows()