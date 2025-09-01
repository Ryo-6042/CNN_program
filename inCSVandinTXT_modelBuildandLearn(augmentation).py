import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import os
import time  # 学習時間測定のため追加
import csv

import re

from tensorflow.keras.optimizers import Adam  # 5/28追加

# フォルダのパス
train_dir = r"C:\Users\nakaz_fls170b\Desktop\kuma\Datasets_0_30000_3_500_0.01\train"
answer_dir = r"C:\Users\nakaz_fls170b\Desktop\kuma\Datasets_0_30000_3_500_0.01\answer"
heatmap_dir = r"C:\Users\nakaz_fls170b\Desktop\kuma\Datasets_0_30000_5_500_0.0001_3×3\Conv_heatmap"

# 保存用ディレクトリ作成
os.makedirs(heatmap_dir, exist_ok=True)

# CSVファイルをリストにまとめて読み込む
train_files = [
    os.path.join(train_dir, f)
    for f in os.listdir(train_dir)
    if f.endswith(".csv") and (
        f.startswith("image_data_") or f.startswith("updated_image_data_")
    )
]
train_data = [pd.read_csv(file, header=None) for file in train_files]
train_data = pd.concat(train_data, ignore_index=True)

# TXTファイルをリストにまとめて読み込む
answer_files = [os.path.join(answer_dir, f) for f in os.listdir(answer_dir)
                if f.startswith("frame_") or f.startswith("generated_")]
answer_data = [pd.read_csv(file, header=None) for file in answer_files]
answer_data = pd.concat(answer_data, ignore_index=True)

# データの前処理
X_train = np.array(train_data).reshape(-1, 50, 50, 1)  # 必要に応じてリサイズ
y_train = np.array(answer_data)

# CNNモデルの構築
input_shape = (50, 50, 1)

# 畳み込み層のパラメータ設定（必要に応じて変更してください）
kernel_size = (9, 9)
stride_size = (1, 1)  # 例：1ピクセルずつスライド

padding_type = 'same'  # or 'valid'

model = models.Sequential([
    layers.Conv2D(32, kernel_size=kernel_size, strides=stride_size, activation='relu', padding=padding_type, input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, kernel_size=kernel_size, strides=stride_size, activation='relu', padding=padding_type),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, kernel_size=kernel_size, strides=stride_size, activation='relu', padding=padding_type),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, kernel_size=kernel_size, strides=stride_size, activation='relu', padding=padding_type),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(512, kernel_size=kernel_size, strides=stride_size, activation='relu', padding=padding_type),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# 学習率の設定
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# 学習時間の測定開始
start_time = time.time()

# モデルの学習
history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2)

# 学習時間の測定終了
end_time = time.time()
training_time = end_time - start_time
print(f"Total training time: {training_time:.2f} seconds")

# 学習結果の可視化
epochs = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'], label='Training Loss')
plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 学習結果をCSVファイルに保存
save_dir = r"C:\Users\nakaz_fls170b\Desktop\kuma\Datasets_0_15000_5_500_0.0001_9×9\training_result"
os.makedirs(save_dir, exist_ok=True)

train_results_file = os.path.join(save_dir, "train_results.csv")

headers = ["Epoch", "Train Accuracy", "Validation Accuracy", "Train Loss", "Validation Loss"]
epochs_data = list(zip(epochs, history.history["accuracy"], history.history["val_accuracy"],
                       history.history["loss"], history.history["val_loss"]))

with open(train_results_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(epochs_data)

print(f"学習結果を {train_results_file} に保存しました！")

# 学習モデルの保存
model.save("cnn_model_15000_5_500_0.0001_9×9.h5")

# モデル情報の保存
model_summary_path = r"C:\Users\nakaz_fls170b\Desktop\kuma\Datasets_0_15000_5_500_0.0001_9×9\model\model_summary.txt"
with open(model_summary_path, "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))
    f.write(f"\nTotal training time: {training_time:.2f} seconds\n")

# 特徴マップの可視化（CSVファイルの順番で処理）
def visualize_feature_maps(model, X_sample, filename):
    layer_outputs = [layer.output for layer in model.layers if "conv" in layer.name]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(X_sample.reshape(1, 50, 50, 1))

    plt.figure(figsize=(12, 4))
    
    # 元画像
    plt.subplot(1, len(activations) + 1, 1)
    plt.imshow(X_sample.squeeze(), cmap="gray")
    plt.title("Original Image")

    # 畳み込み層の特徴マップを順番に表示
    for i, activation in enumerate(activations):
        plt.subplot(1, len(activations) + 1, i + 2)
        plt.imshow(activation[0, :, :, 0], cmap="viridis")
        plt.title(f"Conv Layer {i+1}")
    heatmap_filename = os.path.join(heatmap_dir, f"{filename}_heatmap.png")
    plt.savefig(heatmap_filename)
    plt.close()


# 特定ファイルのみ処理
# 比較対象の数字（文字列でも整数でもOK）
target_numbers = {"0", "6", "7"}

# ファイル名の末尾の数字を抽出して比較
for i, X_sample in enumerate(X_train):
    csv_filename = os.path.basename(train_files[i]).replace(".csv", "")
    match = re.search(r'\d+$', csv_filename)  # 末尾の数字を抽出

    if match and match.group() in target_numbers:
        visualize_feature_maps(model, X_sample, csv_filename)


# CSVファイル順で特徴マップを生成・保存
# for i, X_sample in enumerate(X_train):
#     csv_filename = os.path.basename(train_files[i]).replace(".csv", "")
#     visualize_feature_maps(model, X_sample, csv_filename)

print(f"Feature maps saved in {heatmap_dir}")

print(f"Model summary saved to {model_summary_path}")
