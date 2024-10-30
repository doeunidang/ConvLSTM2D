# evaluate.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import load_test_data, expand_mask_to_batch_size
import matplotlib.pyplot as plt
import os

# 마스크된 MSE 손실 함수
def masked_mse(y_true, y_pred, mask):
    mask = tf.cast(mask, dtype=tf.float32)
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_sum(squared_difference * mask) / tf.reduce_sum(mask)

# 모델 로드 함수
def load_trained_model(model_path):
    return load_model(
        model_path,
        custom_objects={'masked_mse': masked_mse},
        compile=False
    )

# Junction 위치에서만 flooding 값을 예측하고 시각화하는 함수
def plot_prediction(y_true, y_pred, junction_locations, index=0, save_path="prediction.png"):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(y_true[index].reshape(64, 64), cmap='Blues')
    plt.title('True Flooding Output')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(y_pred[index].reshape(64, 64), cmap='Blues')
    plt.title('Predicted Flooding Output')
    plt.colorbar()
    
    plt.savefig(save_path)
    plt.close()

# 평가 함수
def evaluate():
    model_path = '/content/ConvLSTM2D/convlstm_model.keras'
    model = load_trained_model(model_path)
    
    # 데이터 로드
    X_test, y_test, test_masks = load_test_data()
    
    # 모델 컴파일
    model.compile(
        optimizer='adam',
        loss=lambda y_true, y_pred: masked_mse(
            y_true, y_pred, expand_mask_to_batch_size(test_masks[0], tf.shape(y_true)[0])
        ),
        metrics=['mae']
    )
    
    # 평가 및 예측
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Loss (MSE): {test_loss}")
    print(f"Test MAE: {test_mae}")
    
    # 예측 결과 확인
    predictions = model.predict(X_test)
    terrain_data_path = '/content/ConvLSTM2D/DATA_numpy/terrain_data.npy'
    terrain_data = np.load(terrain_data_path, allow_pickle=True)
    junction_locations = {str(terrain_data[i]['Junction']): (i // 64, i % 64) for i in range(terrain_data.size) if terrain_data[i]['Junction']}
    
    # 예측 결과 저장
    plot_prediction(y_test, predictions, junction_locations, index=0, save_path="prediction.png")
    print("Prediction saved as 'prediction.png'.")

if __name__ == "__main__":
    evaluate()
