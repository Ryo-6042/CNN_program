import nidaqmx
import time
import csv
from datetime import datetime
import cv2
import os
import sys

# --- 設定値 ---

# Dev2\ao0の出力電圧（電気泳動用）
ao0_on_voltage = 2.1  # 堆積時の印加電圧

# Dev2\ao1（PZT）の電圧範囲
z_min_voltage = 0.0
z_max_voltage = 4.99

# Dev3\ao1の出力電圧（XYステージ用)
x_min = 0.0
x_max = 0.0

y_min = 0.0
y_max = 0.0

# ao1の電圧と変位の関係
VOLT_TO_UM = 6.6815 # 1Vあたり6.6815µm変位

# 事前に電圧をかけてPZTの遊びを無くすためのプリロード電圧
ao1_preload_voltage = 0.5

# アナログ出力の変化速度（秒）
change_rate = 0.2  # 堆積準備・初期化用
change_rate_deposition = 0.0121  # 堆積用

# 電圧変化量（1ステップあたり）
voltage_rate = 0.0017

# --- カメラ・録画関連設定 ---
CAMERA_ID = 0
SAVE_DIR = r"C:\Users\nakaz_fls170b\Desktop\kuma\video_recordings" # 録画パス
FPS = 30.0

# ★★★ 修正点: ROI座標は [左端x, 上端y, 右端x, 下端y] の順で指定します ★★★
# 元の座標: [300, 350, 195, 245] -> 幅と高さがマイナスになりエラー
# 正しい座標の例:
ROI_COORDS = [195, 245, 350, 400] 

# --- リソース変数の初期化 ---
cap = None
video_writer = None
task = None
digital_task = None

try:
    # --- 事前計算 ---
    length = (z_max_voltage - ao1_preload_voltage) * VOLT_TO_UM
    time_build_pillar = (z_max_voltage - ao1_preload_voltage) * change_rate_deposition / voltage_rate
    velocity = (length / time_build_pillar) * 1000 if time_build_pillar > 0 else 0

    print(f"ピラーの長さ about： {length:.2f} µm, 堆積時間 about： {time_build_pillar:.2f} s")
    print(f"堆積速度 about： {velocity:.2f} nm/s")
    print("-" * 30)

    # --- 保存ディレクトリの作成 ---
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"動画は '{SAVE_DIR}' ディレクトリに保存されます。")

    # --- カメラの初期化 ---
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise IOError(f"エラー: カメラID {CAMERA_ID} を開けませんでした。")

    # --- 録画ファイルの設定 ---
    x1, y1, x2, y2 = ROI_COORDS
    # ROIの幅と高さが正であることを確認
    if not (x1 < x2 and y1 < y2):
        raise ValueError(f"ROIの座標指定が不正です。[x1, y1, x2, y2]の形式で、x1 < x2 かつ y1 < y2 となるようにしてください。 現在値: {ROI_COORDS}")
    
    roi_width = x2 - x1
    roi_height = y2 - y1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_filename = os.path.join(SAVE_DIR, f"rec_roi_{current_time}.mp4")

    video_writer = cv2.VideoWriter(video_filename, fourcc, FPS, (roi_width, roi_height))
    if not video_writer.isOpened():
        raise IOError(f"エラー: 録画ファイルを作成できませんでした。パス: {video_filename}")
    print(f"カメラ準備完了。ROIを録画します: {video_filename}")

    # --- DAQタスクの初期化 ---
    # --- PZT印加電圧と基板への印加電圧用 ---
    task = nidaqmx.Task()
    task.ao_channels.add_ao_voltage_chan("Dev2/ao0", min_val=0.0, max_val=5.0)
    task.ao_channels.add_ao_voltage_chan("Dev2/ao1", min_val=0.0, max_val=5.0)
    task.start()
    
    digital_task = nidaqmx.Task()
    digital_task.do_channels.add_do_chan("Dev2/port0/line0")
    digital_task.start()

    # --- XY印加電圧 ---
    task2 = nidaqmx.Task()
    task2.ao_channels.add_ao_voltage_chan("Dev3/ao0", min_val=0.0, max_val=5.0)
    task2.ao_channels.add_ao_voltage_chan("Dev3/ao1", min_val=0.0, max_val=5.0)
    task2.start()

    digital_task2 = nidaqmx.Task()
    digital_task2.do_channels.add_do_chan("Dev3/port0/line0")
    digital_task2.start()

    # --- 初期状態設定 ---
    z_voltage = 0.0
    ao0_voltage = 0.0
    task.write([ao0_voltage, z_voltage])
    digital_task.write(False)
    print("タスクを開始し、初期電圧を [0.0, 0.0] に設定しました。")
    time.sleep(0.2)

    # --- プリロード電圧印加 ---
    if input("堆積準備（プリロード）を実行しますか？ (yes/no): ").lower() == "yes":
        while z_voltage < ao1_preload_voltage:
            z_voltage = min(z_voltage + 0.05, ao1_preload_voltage)
            task.write([ao0_voltage, z_voltage])
            print(f"プリロード中... Voltage: ao0={ao0_voltage:.2f}V, ao1={z_voltage:.4f}V")
            time.sleep(change_rate)
        print(f"プリロード完了. Voltage: ao0={ao0_voltage:.2f}V, ao1={z_voltage:.4f}V")
    else:
        print("プリロードは実行されませんでした。")
        z_voltage = ao1_preload_voltage # スキップしても電圧はプリロード値に
        task.write([ao0_voltage, z_voltage])

    print("-" * 30)

    # --- 堆積実行 ---
    if input("堆積を実行しますか？ (yes/no): ").lower() == "yes":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"deposition_log_{timestamp}.csv"
        print(f"ログファイル: {log_filename}")

        with open(log_filename, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['time(s)', 'pzt_voltage(V)', 'pillar_length(um)'])
            
            # デジタル出力と電気泳動電圧をON
            digital_task.write(True)
            ao0_voltage = ao0_on_voltage
            task.write([ao0_voltage, z_voltage])
            print("電気泳動を開始しました。2秒待機します...")
            time.sleep(2)

            start_time = time.time()
            print("堆積を開始します...'q'キーで中断できます。")
            
            # ★★★ 修正点: 堆積ループの中で映像取得と録画を行う ★★★
            while z_voltage <= z_max_voltage:
                # --- 映像処理 ---
                ret, frame = cap.read()
                if ret:
                    # ROI部分を切り出して録画
                    roi_frame = frame[y1:y2, x1:x2]
                    video_writer.write(roi_frame)
                    
                    # 表示用のフレームにROIの枠を描画
                    display_frame = frame.copy()
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 映像を表示
                    cv2.imshow('CCD Camera Feed (Full View with ROI)', display_frame)
                    cv2.imshow('Recorded ROI', roi_frame)

                # 'q'キーで中断
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("'q'キーが押されたため、処理を中断します。")
                    break

                # --- 電圧更新とログ記録 ---
                task.write([ao0_voltage, z_voltage])
                
                elapsed_time = time.time() - start_time
                current_length = max(0, (z_voltage - ao1_preload_voltage) * VOLT_TO_UM)
                
                csv_writer.writerow([f"{elapsed_time:.4f}", f"{z_voltage:.4f}", f"{current_length:.4f}"])
                print(f"\r堆積中... Time: {elapsed_time:6.2f}s, Voltage: {z_voltage:6.4f}V, Length: {current_length:6.3f}µm", end="")

                z_voltage += voltage_rate
                time.sleep(change_rate_deposition)

            print("\n堆積完了.")
            digital_task.write(False)
            print("デジタル出力をOFFにしました。")

    else:
        print("堆積はキャンセルされました。")

    print("-" * 30)

    # --- 初期化 ---
    if input("電圧を初期値(0V)に戻しますか？ (yes/no): ").lower() == "yes":
        ao0_voltage = 0.0
        while z_voltage > z_min_voltage:
            z_voltage = max(z_voltage - 0.05, z_min_voltage)
            task.write([ao0_voltage, z_voltage])
            print(f"初期化中... Voltage: ao0={ao0_voltage:.2f}V, ao1={z_voltage:.4f}V")
            time.sleep(change_rate)
        print(f"初期化完了. Voltage: ao0={ao0_voltage:.2f}V, ao1={z_min_voltage:.4f}V")

except (nidaqmx.errors.DaqError, IOError, ValueError, KeyboardInterrupt) as e:
    print(f"\nエラーまたは中断が発生しました: {e}")
finally:
    # --- リソース解放 ---
    print("\nリソースを解放しています...")
    if cap and cap.isOpened():
        cap.release()
    if video_writer and video_writer.isOpened():
        video_writer.release()
    cv2.destroyAllWindows()
    
    if digital_task:
        digital_task.write(False) # 念のためOFFに
        digital_task.close()
    if task:
        task.write([0.0, 0.0]) # 安全のため0Vに
        task.close()
        
    print("すべてのリソースを解放しました。プログラムを終了します。")

