import cv2

# --- 設定項目 ---
# 切り取る範囲(ROI)の座標 (左上x1,y1, 右下x2,y2)
x1, y1 = 250, 150
x2, y2 = 400, 300

# ★拡大率をパーセントで指定 (100 = 等倍, 200 = 2倍)
zoom_percentage = 200

# 枠の色 (B, G, R) と太さ
roi_color = (0, 255, 0) # 緑色
roi_thickness = 2      # 太さ 2px
# ---------------------------------------------

# カメラのキャプチャを開始 (0はデフォルトカメラ, 1は2台目など)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("エラー: カメラを開けませんでした。")
    exit()

print("カメラ映像を表示します。ウィンドウを選択した状態で 'q' キーを押すと終了します。")

while True:
    ret, frame = cap.read()
    if not ret:
        print("エラー: フレームを読み込めませんでした。")
        break

    # 1. 元の映像から指定した座標で範囲(ROI)を切り取る
    roi_frame = frame[y1:y2, x1:x2]

    # 切り取ったROIが空でないか確認
    if roi_frame.size == 0:
        continue # 空の場合は以降の処理をスキップ

    # 2. ★切り取ったROIを、指定したパーセンテージで拡大する
    # 元のROIの高と幅を取得
    height, width = roi_frame.shape[:2]
    # 指定パーセンテージから新しいサイズを計算
    new_width = int(width * zoom_percentage / 100)
    new_height = int(height * zoom_percentage / 100)
    # cv2.resizeで画像サイズを変更
    zoomed_roi = cv2.resize(roi_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # 3. 元の映像にROIを示す四角い枠を描画する
    cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, roi_thickness)

    # 4. 元映像と「拡大したROI」をそれぞれ表示する
    cv2.imshow('Original Feed (with ROI box)', frame)
    cv2.imshow(f'Zoomed ROI ({zoom_percentage}%)', zoomed_roi)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後片付け
cap.release()
cv2.destroyAllWindows()

print("プログラムを終了しました。")
