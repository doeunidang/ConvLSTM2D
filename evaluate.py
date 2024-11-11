import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import load_test_data
from losses import custom_loss
from preprocess import load_shapefile  # Junction 위치와 이름 로드 함수
import os

# 모델 및 데이터 경로 설정
model_path = '/content/ConvLSTM2D/model/convlstm_model.keras'
shapefile_path = '/content/ConvLSTM2D/DATA_input/DEM/DEM_GRID.shp'
output_folder = '/content/ConvLSTM2D/results'

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
    
    # 각 Junction에서의 True/Predicted 값 출력
    for row, col, junction_name in junction_indices:
        print(f"\n{junction_name} (위치: [{row}, {col}])")
        for i, t in enumerate(time_steps):
            true_value = y_test[sample_idx, t, row, col, 0]
            pred_value = y_pred[sample_idx, t, row, col, 0]
            print(f"{times[i]} - True: {true_value:.4f}, Predicted: {pred_value:.4f}")
    
    # 선택한 sample_idx에 해당하는 특정 셀의 rainfall 값 출력
    row, col = cell_position
    print(f"\nRainfall Values at Position {cell_position} for Selected Sample:")
    for i, t in enumerate(time_steps):
        rainfall_value = X_test[sample_idx, t, row, col, 0]  # 특정 셀의 rainfall 값
        print(f"{times[i]} - Rainfall at {cell_position}: {rainfall_value:.4f}")

def evaluate():
    """모델 평가 및 결과 시각화 함수."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 모델과 테스트 데이터 로드
    model = load_model(model_path, custom_objects={'custom_loss': custom_loss})
    X_test, y_test = load_test_data()  # (배치, 4, 64, 64, 5), (배치, 4, 64, 64, 1)
    
    # Junction 위치와 이름 로드
    _, _, junction_indices = load_shapefile(shapefile_path)  # junction_indices에 (row, col, 이름) 포함
    
    # 모델 예측 수행
    y_pred = model.predict(X_test)  # 예측 결과 (배치, 4, 64, 64, 1)
    
    # 첫 번째 테스트 샘플의 예측 결과 시각화 및 Junction 비교 출력
    plot_and_compare_results(X_test, y_test, y_pred, junction_indices, sample_idx=1, cell_position=(32, 32))

if __name__ == "__main__":
    evaluate()
