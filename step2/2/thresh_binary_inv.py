import cv2

img=cv2.imread('../../sample/frame/sample_video1_gray_0100.jpg',cv2.IMREAD_GRAYSCALE)
th,dst=cv2.threshold(img,115,255,cv2.THRESH_BINARY_INV)
#cv2.threshold(src,thresh,maxValue,thresholoType)
#src：インプットの画像（白黒）
#thresh：閾値となる値
#maxValue：閾値条件を満たした場合の値
#thresholdType：閾値処理のタイプ

cv2.imshow("threshold_inv.jpg",dst)

k=cv2.waitKey()
if k==ord('q'):
    cv2.destroyAllWindows()
elif k==ord('s'):
    cv2.imwrite('../../sample/frame/frame_practice/thresholo_inv.jpg',dst)
    cv2.destroyAllWindows()