import cv2
import numpy as np

img=cv2.imread('../../sample/frame/sample_video1_gray_0100.jpg')

#マーカの位置を指定
marker_positions=np.array([[50,100],[200,200],[50,200]],dtype=np.float32)

#マーカを描画
for marker_pos in marker_positions:
    cv2.drawMarker(img,tuple(marker_pos.astype(int)),(0,255,0),markerType=cv2.MARKER_CROSS,markerSize=30,thickness=10)

#アフィン変換前の座礁を指定
src_poitns=marker_positions

#アフィン変換後の座標を指定
dst_points=np.array([[10,100],[200,50],[100,250]],dtype=np.float32)

#アフィン変換行列を計算
M= cv2.getAffineTransform(src_poitns,dst_points)

#アフィン変換を適用
affine_image=cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))

#画像表示
cv2.imshow("Img with markers and Affine Transformation",affine_image)

k=cv2.waitKey()
if k==ord('q'):
    cv2.destroyAllWindows()
elif k==ord("s"):
    cv2.imwrite('../../sample/frame/affine_img.jpg',affine_image)
    cv2.destroyAllWindows()