import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import load_test_data
from losses import custom_loss
from preprocess import load_shapefile
import tensorflow as tf
import os

# 모델 및 데이터 경로 설정
model_path = '/content/ConvLSTM2D/model/convlstm_model.keras'
shapefile_path = '/content/ConvLSTM2D/DATA_input/DEM/DEM_GRID.shp'
output_folder = '/content/ConvLSTM2D/results'

def load_junction_mask(junction_mask_path='/content/ConvLSTM2D/DATA_numpy/junction_mask.npy'):
    """
    Junction mask를 로드하고 배치/시간/채널 차원을 추가합니다.
    """
    junction_mask = np.load(junction_mask_path)  # (64, 64, 1)
    junction_mask = tf.convert_to_tensor(junction_mask, dtype=tf.float32)
    junction_mask = tf.expand_dims(junction_mask, axis=0)  # 배치 차원 추가
    junction_mask = tf.expand_dims(junction_mask, axis=0)  # 시간 차원 추가
    junction_mask = tf.expand_dims(junction_mask, axis=-1)  # 채널 차원 추가
    return junction_mask  # (1, 1, 64, 64, 1)

@tf.keras.utils.register_keras_serializable(package="Custom", name="custom_accuracy")
def custom_accuracy(y_true, y_pred):
    """
    유출량 예측에서 상대 오차를 기준으로 정확도를 계산합니다.
    """
    junction_mask = load_junction_mask()
    batch_size = tf.shape(y_pred)[0]
    time_steps = tf.shape(y_pred)[1]
    junction_mask_broadcasted = tf.tile(junction_mask, [batch_size, time_steps, 1, 1, 1])

    y_true_masked = y_true * junction_mask_broadcasted
    y_pred_masked = y_pred * junction_mask_broadcasted

    relative_error = tf.abs(y_true_masked - y_pred_masked) / (y_true_masked + tf.keras.backend.epsilon())
    within_tolerance = relative_error <= 0.2  # 허용 상대 오차 (20%)

    correct_predictions = tf.reduce_sum(tf.cast(within_tolerance, tf.float32) * junction_mask_broadcasted, axis=[2, 3, 4])
    total_junctions = tf.reduce_sum(junction_mask_broadcasted, axis=[2, 3, 4])

    accuracy_per_frame = correct_predictions / (total_junctions + tf.keras.backend.epsilon())
    mean_accuracy = tf.reduce_mean(accuracy_per_frame)

    return mean_accuracy

def plot_and_compare_results(X_test, y_test, y_pred, junction_indices, sample_idx=0, cell_position=(32, 32)):
    """예측 결과와 실제 결과를 비교하여 시각화 및 Junction 위치의 True/Predicted 값을 출력합니다."""
    time_steps = range(4)
    times = ["t-20", "t-10", "t", "t+10"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for i, t in enumerate(time_steps):
        actual = y_test[sample_idx, t].squeeze()  # 실제 결과 (4, 64, 64, 1)
        predicted = y_pred[sample_idx, t].squeeze()  # 예측 결과 (4, 64, 64, 1)
        
        # 실제 결과 시각화
        ax = axes[0, i]
        ax.imshow(actual, cmap='Blues', vmin=0, vmax=np.nanmax(actual))
        ax.set_title(f'Actual Flooding ({times[i]})')
        
        # 예측 결과 시각화
        ax = axes[1, i]
        ax.imshow(predicted, cmap='Blues', vmin=0, vmax=np.nanmax(predicted))
        ax.set_title(f'Predicted Flooding ({times[i]})')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'flooding_predictions.png'))
    plt.show()
    
    for row, col, junction_name in junction_indices:
        print(f"\n{junction_name} (위치: [{row}, {col}])")
        for i, t in enumerate(time_steps):
            true_value = y_test[sample_idx, t, row, col, 0]
            pred_value = y_pred[sample_idx, t, row, col, 0]
            print(f"{times[i]} - True: {true_value:.4f}, Predicted: {pred_value:.4f}")

    row, col = cell_position
    print(f"\nRainfall Values at Position {cell_position} for Selected Sample:")
    for i, t in enumerate(time_steps):
        rainfall_value = X_test[sample_idx, t, row, col, 0]
        print(f"{times[i]} - Rainfall at {cell_position}: {rainfall_value:.4f}")

def evaluate():
    """모델 평가 및 결과 시각화 함수."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    model = load_model(
        model_path,
        custom_objects={'custom_loss': custom_loss, 'custom_accuracy': custom_accuracy}
    )
    X_test, y_test = load_test_data()
    _, _, junction_indices = load_shapefile(shapefile_path)

    y_pred = model.predict(X_test)

    plot_and_compare_results(X_test, y_test, y_pred, junction_indices, sample_idx=1, cell_position=(32, 32))

if __name__ == "__main__":
    evaluate()
