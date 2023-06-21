import cv2

img=cv2.imread('../../sample/frame/sample.jpg')

th,dst=cv2.threshold(img,120,255,cv2.THRESH_TOZERO)
cv2.imshow("thresh-tozero.jpg",dst)

k=cv2.waitKey()
if k==ord('q'):
    cv2.destroyAllWindows()
elif k==ord('s'):
    cv2.imwrite('../../sample/frame/thresh_img/thresh_tozero.jpg')
