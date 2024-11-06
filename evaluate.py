import numpy as np
from tensorflow.keras.models import load_model
from utils import load_test_data
import matplotlib.pyplot as plt
import os
from preprocess import load_shapefile
from losses import custom_loss

def load_trained_model(model_path):
    """저장된 모델을 로드하여 반환합니다."""
    model = load_model(model_path, custom_objects={"custom_loss": custom_loss})
    return model

def evaluate_model(model, X_test, y_test, junction_mask):
    """
    테스트 데이터에 대해 모델을 평가하고 각 타임스텝별 예측 및 실제 값을 계산합니다.
    """
    y_pred_series = model.predict(X_test)
    
    # 예측 값과 실제 값의 저장할 리스트
    y_preds, y_trues = [], []
    time_steps = y_pred_series.shape[1]  # 타임스텝에 따른 예측 개수

    for t in range(time_steps):
        # 각 타임스텝에서 mask를 이용하여 필요한 값 추출
        mask = np.logical_not(np.isnan(y_test[:, t]))  # 현재 타임스텝에서 유효한 값만 선택
        
        # y_test와 y_pred_series의 현재 타임스텝 데이터를 64x64 형태로 맞추기
        y_true_t = y_test[:, t].reshape(-1, 64, 64)
        y_pred_t = y_pred_series[:, t].reshape(-1, 64, 64)
        
        y_trues.append(y_true_t)
        y_preds.append(y_pred_t)

    return y_preds, y_trues

def plot_and_save_prediction(y_trues, y_preds, output_folder='/content/ConvLSTM2D/results'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 64x64에서 30x30 영역의 시작 인덱스
    start_row = 17  # 64의 중앙에서 30x30 영역을 가져오기 위해
    start_col = 17

    for idx, (y_true, y_pred) in enumerate(zip(y_trues, y_preds)):
        minutes_ahead = (idx + 1) * 10
        plt.figure(figsize=(10, 5))
        
        # 첫 번째 샘플에 대한 시각화
        plt.subplot(1, 2, 1)
        plt.imshow(y_true[0], cmap='Blues', vmin=0, vmax=np.nanmax(y_true))
        plt.title(f'True Flooding Output ({minutes_ahead} Minutes Ahead)')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(y_pred[0], cmap='Blues', vmin=0, vmax=np.nanmax(y_pred))
        plt.title(f'Predicted Flooding Output ({minutes_ahead} Minutes Ahead)')
        plt.colorbar()
        
        output_path = os.path.join(output_folder, f'prediction_{minutes_ahead}min.png')
        plt.savefig(output_path)
        plt.close()

        print(f"Prediction image for {minutes_ahead} minutes ahead saved at {output_path}")

def inspect_predictions(y_trues, y_preds, junction_indices, actual_flooding_file, actual_rainfall_file):
    print(f"\n사용된 rainfall 파일 (입력): {actual_rainfall_file}")
    print(f"사용된 flooding 파일 (실제 값): {actual_flooding_file}\n")
    
    # 64x64에서 10x10 영역의 시작 인덱스
    start_row = 17  # 64의 중앙에서 10x10 영역을 가져오기 위해
    start_col = 17

    for idx, (y_true, y_pred) in enumerate(zip(y_trues, y_preds)):
        minutes_ahead = (idx + 1) * 10
        print(f"\nTime Step ({minutes_ahead} Minutes Ahead)")

        # 10x10 영역의 실제값과 예측값 출력
        print("\n10x10 영역의 true값과 예측값:")
        true_values = y_true[0][start_row:start_row + 10, start_col:start_col + 10]
        pred_values = y_pred[0][start_row:start_row + 10, start_col:start_col + 10]

        for row in range(10):
            for col in range(10):
                true_value = true_values[row, col]
                pred_value = pred_values[row, col]
                print(f"Location ({start_row + row}, {start_col + col}) -> True: {true_value:.4f}, Predicted: {pred_value:.4f}")

        print("\n각 junction 위치의 true값과 예측값:")
        for j_idx, (row, col, _) in enumerate(junction_indices):
            true_value = y_true[0, row, col]  # 첫 번째 샘플의 특정 위치 값
            pred_value = y_pred[0, row, col]  # 첫 번째 샘플의 특정 위치 값
            print(f"Junction {j_idx + 1} (위치: [{row}, {col}]) -> True: {true_value:.4f}, Predicted: {pred_value:.4f}")

def evaluate():
    model_path = '/content/ConvLSTM2D/model/convlstm_model.keras'
    model = load_trained_model(model_path)
    X_test, y_test = load_test_data()
    junction_mask = np.load("/content/ConvLSTM2D/DATA_numpy/junction_mask.npy")
    predictions, y_trues = evaluate_model(model, X_test, y_test, junction_mask)
    
    rainfall_file = '/content/ConvLSTM2D/DATA_input/RAINFALL/rainfall_event_251.dat'
    flooding_file = '/content/ConvLSTM2D/DATA_goal/Junction_Flooding_251.xlsx'

    plot_and_save_prediction(y_trues, predictions)
    _, junction_mask, junction_indices = load_shapefile("/content/ConvLSTM2D/DATA_input/DEM/DEM_GRID.shp")
    inspect_predictions(y_trues, predictions, junction_indices, flooding_file, rainfall_file)

if __name__ == "__main__":
    evaluate()
