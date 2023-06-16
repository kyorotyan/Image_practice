import cv2
import numpy as np

img=cv2.imread('/Users/kouki/Image_practice/sample/frame/sample_video1_gray_0100.jpg')
#　並進行列の作成
tx=100 #x方向の移動量
ty=100 #y方向の移動量
tranlation_matrix= np.float32([[1,0,tx],[0,1,ty]])

#画像の移動
translated_image=cv2.warpAffine(img,tranlation_matrix,(img.shape[1],img.shape[0]))

cv2.imshow("translation",translated_image)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('../../sample/frame/translation.png',translated_image)
    cv2.destroyAllWindows()