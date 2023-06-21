import cv2

img=cv2.imread('../../sample/frame/sample.jpg')
th,dst=cv2.threshold(img,120,255,cv2.THRESH_TRUNC)

cv2.imshow("threshold_trunc.jpg",dst)

k=cv2.waitKey()
if k==ord('q'):
    cv2.destroyAllWindows()
elif k==ord('s'):
    cv2.imwrite('../../sample/frame/frame_practice/thresholo_trunc.jpg',dst)
    cv2.destroyAllWindows()