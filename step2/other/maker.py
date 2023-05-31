import cv2

# 画像読み込み
image = cv2.imread('/Users/kouki/Image_practice/sample/frame/sample_video1_gray_0100.jpg')

# マーカの位置を指定
marker_positions = [(100, 100), (1900, 100), (100, 1900)]  # 複数の座標をリストで指定

# マーカを描画
for (marker_x, marker_y) in marker_positions:
    cv2.drawMarker(image, (marker_x, marker_y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=30, thickness=10)

# 画像表示
cv2.imshow('Image with Markers', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
