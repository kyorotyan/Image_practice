import cv2
import numpy as np

# 動画ファイルのパスを指定
video_path = '/Users/kouki/Image_practice/sample/image1.avi'

# 動画を開くためのキャプチャオブジェクトを作成
cap = cv2.VideoCapture(video_path)

# キャプチャオブジェクトが正常に初期化されているかチェック
if not cap.isOpened():
    print("動画を開けませんでした。")
    exit()

# フレームを連続して読み込み、表示する
while True:
    # 1フレームずつ読み込む
    ret, frame = cap.read()

    # 読み込みが成功したかどうかをチェック
    if not ret:
        print("動画の読み込みが完了しました。")
        break

    # フレームを表示
    cv2.imshow('Video', frame)

    # 'q'キーが押されたら終了する
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャオブジェクトとウィンドウを解放
cap.release()
cv2.destroyAllWindows()
