import cv2
import numpy as np

img=cv2.imread('../../sample/frame/scaling.png')

height,width,channels=img.shape[:3]

source_poitns=np.array([[0,0],[0,height],[width,height],[width,0]],dtype=np.float32)
target_poitns=np.array([[200,0],[0,600],[600,600],[400,0]],dtype=np.float32)

M = cv2.getPerspectiveTransform(source_poitns,target_poitns)

persective_image=cv2.warpPerspective(img,M,(width,height))

cv2.imshow("sample",persective_image)

k=cv2.waitKey()
if k== ord('q'):
    cv2.destroyAllWindows
elif k==ord('s'):
    cv2.imwrite('../../sample/frame/ImageSpective.jpg',persective_image)