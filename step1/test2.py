import cv2
import numpy as np

# 動画ファイルのパスを指定
video_path = 'sample_video1.avi'

# 取得するフレームの番号を指定
target_frame_number = 100  # 例として100番目のフレームを取得する

# 出力ファイルのパスを指定
output_path = 'frames'

# 動画を開くためのキャプチャオブジェクトを作成
cap = cv2.VideoCapture(video_path)

# キャプチャオブジェクトが正常に初期化されているかチェック
if not cap.isOpened():
    print("動画を開けませんでした。")
    exit()

# 指定したフレーム番号までフレームを読み込む
current_frame_number = -1
while current_frame_number < target_frame_number:
    ret, frame = cap.read()
    if not ret:
        print("指定したフレームが存在しません。")
        exit()
    current_frame_number += 1

# フレームをNumPy配列に変換
frame_array = np.array(frame)

# フレームのR成分のみを取り出す
frame_r = frame_array[:, :, 2]

# 出力ファイル名を生成
output_filename = f"{video_path.split('/')[-1].split('.')[0]}_gray_{target_frame_number:04d}.jpg"
output_file_path = f"{output_path}/{output_filename}"

# R成分のフレームを保存
cv2.imwrite(output_file_path, frame_r)

# キャプチャオブジェクトとウィンドウを解放
cap.release()
cv2.destroyAllWindows()
