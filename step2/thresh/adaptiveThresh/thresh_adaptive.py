import cv2 
import numpy as np
from matplotlib import pyplot as plt

#画像の読み込み
img=cv2.imread('../../../sample/frame/sample.jpg')

#適応的閾値処理による二値化
thresh=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#第1引数：二値化する画像データ。
#第2引数：輝度値の最大値。
#第3引数：閾値計算のタイプ。
#「ADAPTIVE_THRESHOLD_GAUSSIAN_C」は局所領域で閾値を計算する方法にガウス分布による重み付けを行った平均値を採用します。
#「ADAPTIVE_THRESHOLD_MEAN_C」にすることで閾値に局所領域の中央値を採用します。
#第4引数：閾値処理の種類。
#第5引数：局所領域のサイズ（奇数である必要がある）。
#第6引数：閾値から引く値。

#結果の表示
cv2.imshow("original Img",img)
cv2.imshow("Adaptive Threshold",thresh)

k=cv2.waitkey()
if k==ord('q'):
    cv2.destroyAllWindows()
elif k==ord('s'):
    cv2.imwrite("../../../sample/frame/thresh_img/adaptive.jpg")
    cv2.destroyAllWindows()