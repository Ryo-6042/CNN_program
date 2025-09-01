# %%
import cv2
# import numpy as np
# import pandas as pd
import nidaqmx
import time
import os
import sys
from tensorflow.keras.models import load_model
from collections import deque
import threading
from ctypes import windll
from datetime import datetime
import csv

# ================================================================
# 1. ユーザー設定項目
# ================================================================

# --- DAQ関連設定 ---
AO_CHANNEL_PZT = "Dev2/ao1"  # PZT伸長用アナログ出力 (Z軸ステージ)
AO_CHANNEL_EPD = "Dev2/ao0"  # 電気泳動用アナログ出力
DO_CHANNEL = "Dev2/port0/line0" # デジタル出力 (シャッター等)

# --- 電圧設定 ---
EPD_VOLTAGE_ON = 2.0          # 電気泳動ON時の電圧
PZT_MAX_VOLTAGE = 1.655       # PZTの最大電圧 (目標長さ)
PZT_MIN_VOLTAGE = 0.0         # PZTの最小電圧
PZT_PRELOAD_VOLTAGE = 0.5     # PZTの遊びをなくすためのプリロード電圧
VOLTAGE_STEP = 0.0017         # 1ステップあたりの電圧変化量
VOLT_TO_UM = 6.6815           # PZT電圧をµmに変換する係数

# --- タイミング設定 ---
DEPOSITION_INTERVAL = 0.0121 # 堆積時の1ステップあたりの待機時間 (秒)
INIT_INTERVAL = 0.2          # 初期化/プリロード時の電圧変化の待機時間 (秒)

# --- カメラ・推論関連設定 ---
CAMERA_ID = 0                 # 使用するカメラのID (通常は0)
# ¥マークは¥¥と2つ重ねるか、先頭にrを付けます
MODEL_PATH = r"C:\Users\pu155197\Desktop\kuma\CNN_program\cnn_model_30000_5_10_0.0001_3×3_1_0809_2.h5" # 読み込む学習済みモデルのパス
# MODEL_PATH = r"C:\Users\pu155197\Desktop\kuma\CNN_program\cnn_model_30000_5_10_0.0001_3×3_1_0809_2.h5" # 読み込む学習済みモデルのパス
# ROI座標の指定方法: [左上x, 左上y, 右下x, 右下y]
ROI_COORDS = [300, 195, 350, 245] 

# --- 映像・ログ保存設定 ---
VIDEO_SAVE_DIR = r"C:\Users\pu155197\Desktop\kuma\video_recordings_detect" # 動画の保存先ディレクトリ
CSV_SAVE_DIR = r"C:\Users\pu155197\Desktop\kuma\csv_recordings_detect" # CSVの保存先ディレクトリ
FPS = 30.0 # 録画のフレームレート

# --- 映像表示設定 ---
ZOOM_PERCENTAGE = 400         # ROIの拡大率 (100 = 等倍, 400 = 4倍)
ROI_BOX_COLOR = (0, 255, 0)   # ROIの枠の色 (B, G, R) - 緑
ROI_BOX_THICKNESS = 1         # ROIの枠の太さ

# --- フィードバック制御設定 ---
JUDGEMENT_HISTORY_LENGTH = 10 # 判定を記録するフレーム数
FAILURE_THRESHOLD = 10        # 堆積中断をトリガーする「×」の閾値

# ================================================================
# 2. クラス・関数定義
# ================================================================

class CameraStream:
    """
    カメラのフレーム読み取りを別スレッドで実行し、メインループの処理をブロックしないようにするクラス。
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        
        if not self.stream.isOpened():
            print(f"エラー: カメラID {src} を開けません。")
            raise ValueError("カメラを開けませんでした。")
            
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.stream.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

def preprocess_for_model(roi_image):
    """ROI画像をモデルの入力形式に前処理する"""
    reshaped_data = roi_image.reshape(1, roi_image.shape[0], roi_image.shape[1], 1)
    return reshaped_data

def predict_deposit(model, roi_image):
    """ROI画像を受け取り、堆積確率を予測する"""
    processed_data = preprocess_for_model(roi_image)
    predictions = model(processed_data, training=False)
    return float(predictions[0][0])

def get_judgement(probability):
    """推論確率から〇×-の判定を返す"""
    if 0.9 <= probability <= 1.0:
        return "〇"
    elif 0.0 <= probability <= 0.1:
        return "×"
    return "-"

# ================================================================
# 3. メイン処理
# ================================================================

def main():
    # --- リソースの初期化 ---
    model = None
    camera_stream = None
    video_writer = None
    windll.winmm.timeBeginPeriod(1)

    try:
        # --- モデルとカメラの準備 ---
        try:
            model = load_model(MODEL_PATH)
            print(f"モデルを読み込みました: {MODEL_PATH}")
        except Exception as e:
            print(f"エラー: モデルの読み込みに失敗しました。パスを確認してください: {MODEL_PATH}\n{e}")
            return

        try:
            camera_stream = CameraStream(CAMERA_ID).start()
            time.sleep(1.0)
            print(f"カメラID {CAMERA_ID} のストリーミングを開始しました。")
        except ValueError:
            return

        # --- DAQタスクと電圧の初期化 ---
        with nidaqmx.Task() as do_task, nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(AO_CHANNEL_EPD, min_val=0.0, max_val=5.0)
            task.ao_channels.add_ao_voltage_chan(AO_CHANNEL_PZT, min_val=0.0, max_val=5.0)
            do_task.do_channels.add_do_chan(DO_CHANNEL)

            pzt_voltage = 0.0
            epd_voltage = 0.0
            task.write([epd_voltage, pzt_voltage], auto_start=True)
            do_task.write(False)

            # --- 堆積準備 (プリロード) ---
            if input("堆積準備（プリロード）を実行しますか？ (yes/no): ").lower() == "yes":
                while pzt_voltage < PZT_PRELOAD_VOLTAGE:
                    pzt_voltage = min(pzt_voltage + 0.05, PZT_PRELOAD_VOLTAGE)
                    task.write([epd_voltage, pzt_voltage], auto_start=True)
                    print(f"プリロード中... 電圧: {pzt_voltage:.3f}V")
                    time.sleep(INIT_INTERVAL)
                print(f"プリロード完了。現在のPZT電圧: {pzt_voltage:.3f}V")
            else:
                print("プリロードはキャンセルされました。")
                return

            # --- 堆積実行 ---
            if input("堆積を実行しますか？ (yes/no): ").lower() == "yes":
                os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)
                os.makedirs(CSV_SAVE_DIR, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = os.path.join(VIDEO_SAVE_DIR, f"rec_{timestamp}.mp4")
                log_filename = os.path.join(CSV_SAVE_DIR, f"log_{timestamp}.csv")
                print(f"録画ファイル: {video_filename}")
                print(f"ログファイル: {log_filename}")

                # ★★★ 変更点1: 録画ファイルのサイズをROIのサイズに設定 ★★★
                x1, y1, x2, y2 = ROI_COORDS
                if not (x1 < x2 and y1 < y2):
                    raise ValueError(f"ROIの座標指定が不正です。 [左上x, 左上y, 右下x, 右下y] の形式で、x1 < x2 かつ y1 < y2 となるようにしてください。 現在値: {ROI_COORDS}")
                
                roi_width = x2 - x1
                roi_height = y2 - y1
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_filename, fourcc, FPS, (roi_width, roi_height))

                if not video_writer.isOpened():
                    raise IOError("エラー: 録画ファイルを開けませんでした。コーデック('mp4v')がシステムにインストールされているか確認してください。")

                with open(log_filename, 'w', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['時間[s]', 'PZT電圧', 'EPV電圧', 'ピラーの長さ[μm]', '推論値', '推論値に対する評価(〇か✕)'])

                    judgement_history = deque(maxlen=JUDGEMENT_HISTORY_LENGTH)
                    
                    epd_voltage = EPD_VOLTAGE_ON
                    do_task.write(True)
                    task.write([epd_voltage, pzt_voltage], auto_start=True)
                    print("電気泳動を開始しました。2秒待機します...")
                    time.sleep(2)
                    
                    start_time = time.time()

                    # メインループ
                    while pzt_voltage < PZT_MAX_VOLTAGE:
                        pzt_voltage += VOLTAGE_STEP
                        
                        task.write([epd_voltage, pzt_voltage], auto_start=True)
                        time.sleep(DEPOSITION_INTERVAL)
                        
                        ret, frame = camera_stream.read()
                        if not ret or frame is None:
                            print("警告: カメラからフレームを取得できませんでした。")
                            continue
                        
                        # --- 映像表示とROIの切り出し ---
                        display_frame = frame.copy()
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), ROI_BOX_COLOR, ROI_BOX_THICKNESS)
                        
                        # ROI部分を切り出す
                        roi_frame = frame[y1:y2, x1:x2]

                        # ★★★ 変更点2: ROIフレームを録画 ★★★
                        if video_writer is not None:
                            video_writer.write(roi_frame)

                        # --- 映像表示 ---
                        if roi_frame.size > 0:
                            new_width = int(roi_frame.shape[1] * ZOOM_PERCENTAGE / 100)
                            new_height = int(roi_frame.shape[0] * ZOOM_PERCENTAGE / 100)
                            zoomed_roi = cv2.resize(roi_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                            cv2.imshow(f'Zoomed ROI ({ZOOM_PERCENTAGE}%)', zoomed_roi)

                        cv2.imshow('Live Camera Feed', display_frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("'q'キーが押されたため、処理を中断します。")
                            break
                        
                        # --- 推論処理 ---
                        gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

                        if gray_frame.size == 0:
                            print("警告: ROIの切り出しに失敗しました。座標を確認してください。")
                            continue

                        probability = predict_deposit(model, gray_frame)
                        judgement = get_judgement(probability)
                        judgement_history.append(judgement)

                        failure_count = judgement_history.count("×")
                        print(f"PZT電圧: {pzt_voltage:.4f}V | 確率: {probability:.3f} | 判定: {judgement} | 失敗カウント: {failure_count}/{JUDGEMENT_HISTORY_LENGTH}")
                        
                        elapsed_time = time.time() - start_time
                        current_length = max(0, (pzt_voltage - PZT_PRELOAD_VOLTAGE) * VOLT_TO_UM)
                        csv_writer.writerow([
                            f"{elapsed_time:.4f}", f"{pzt_voltage:.4f}", f"{epd_voltage:.2f}",
                            f"{current_length:.4f}", f"{probability:.4f}", judgement
                        ])

                        if failure_count >= FAILURE_THRESHOLD:
                            print(f"** 堆積失敗を検知。中断します。 **")
                            break
                    
                    print("\n堆積プロセスが終了しました。")

            else:
                print("堆積がキャンセルされました。")

            # --- 後処理: 電圧初期化 ---
            print("\n後処理を開始します。")
            do_task.write(False)
            print("デジタル出力をOFFにしました。")

            if input("PZTと電気泳動の電圧を0Vに戻しますか？ (Enterで実行/no): ").lower() != "no":
                epd_voltage = 0.0
                while pzt_voltage > PZT_MIN_VOLTAGE:
                    pzt_voltage = max(pzt_voltage - 0.05, PZT_MIN_VOLTAGE)
                    task.write([epd_voltage, pzt_voltage], auto_start=True)
                    print(f"初期化中... PZT電圧: {pzt_voltage:.3f}V")
                    time.sleep(INIT_INTERVAL)
                print("初期化完了。全ての電圧は0Vです。")

    except KeyboardInterrupt:
        print("\n** プログラムが中断されました。リソースを解放します。 **")
    except Exception as e:
        print(f"\n**予期せぬエラーが発生しました: {e}**")
    finally:
        # ★★★ 修正点3: リソース解放処理をここに集約 ★★★
        # このブロックは、プログラムが正常終了、中断、エラーのいずれの場合でも必ず実行されます。
        # if camera_stream:
        #     camera_stream.stop()
        #     print("カメラを解放しました。")
        
        if video_writer:
            video_writer.release()
            print("録画ファイルを閉じました。")

        if camera_stream:
            camera_stream.stop()
            print("カメラを解放しました。")
            
        cv2.destroyAllWindows()
        windll.winmm.timeEndPeriod(1)
        print("プログラムを終了します。")

if __name__ == "__main__":
    main()
