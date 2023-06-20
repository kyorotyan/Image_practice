import cv2

img=cv2.imread('../../sample/frame/sample_video1_gray_0100.jpg',cv2.IMREAD_GRAYSCALE)
#閾値処理
thresholo_value=155 #閾値の値を設定する
max_value=255 #二値化後のピクセル値の最大値を指定

th,dst=cv2.threshold(img,thresholo_value,max_value,cv2.THRESH_BINARY)
cv2.imshow("thresholo.jpg",dst)

k=cv2.waitKey()
if k==ord("q"):
    cv2.destroyAllWindows()
elif k==ord("s"):
    cv2.imwrite('../../sample/frame/thresh_img/thresholo.jpg',dst)
    cv2.destroyAllWindows()
