import numpy as np
from tensorflow.keras.models import load_model
from utils import load_test_data
import matplotlib.pyplot as plt
import os
from preprocess import load_shapefile
from losses import custom_loss

# 저장된 모델을 로드
def load_trained_model(model_path):
    model = load_model(model_path, custom_objects={"custom_loss": custom_loss})
    return model

# 모델 평가 함수
def evaluate_model(model, X_test, y_test, junction_mask):
    mask = np.logical_not(np.isnan(y_test[0]))
    y_pred_series = model.predict(X_test)
    target_time_steps = [0, 1, 2, 3]
    y_preds, y_trues = [], []

    # 각 타임스텝별 예측 및 실제 값 계산
    for t in target_time_steps:
        y_pred_t = np.zeros_like(y_test[t])
        y_pred_t[mask] = y_pred_series[t][mask]
        y_preds.append(y_pred_t)
        y_trues.append(y_test[t])

    return y_preds, y_trues

# 예측 및 실제 값 시각화 및 저장
def plot_and_save_prediction(y_trues, y_preds, output_folder="/content/ConvLSTM2D/results"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for idx, (y_true, y_pred) in enumerate(zip(y_trues, y_preds)):
        minutes_ahead = (idx + 1) * 10
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(y_true.reshape(64, 64), cmap='Blues', vmin=0, vmax=np.nanmax(y_true))
        plt.title(f'True Flooding Output ({minutes_ahead} Minutes Ahead)')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(y_pred.reshape(64, 64), cmap='Blues', vmin=0, vmax=np.nanmax(y_pred))
        plt.title(f'Predicted Flooding Output ({minutes_ahead} Minutes Ahead)')
        plt.colorbar()
        
        output_path = os.path.join(output_folder, f'prediction_{minutes_ahead}min.png')
        plt.savefig(output_path)
        plt.close()

        print(f"Prediction image for {minutes_ahead} minutes ahead saved at {output_path}")

# 예측 값을 검증할 수 있도록 출력
def inspect_predictions(y_trues, y_preds, junction_indices, flooding_file):
    print(f"\nTrue값으로 사용된 flooding 파일 이름: {flooding_file}")
    for idx, (y_true, y_pred) in enumerate(zip(y_trues, y_preds)):
        minutes_ahead = (idx + 1) * 10
        print(f"\nTime Step ({minutes_ahead} Minutes Ahead)")

        if y_true.ndim == 3:
            y_true = y_true[:, :, 0]
        if y_pred.ndim == 3:
            y_pred = y_pred[:, :, 0]

        print("\n각 junction 위치의 true값과 예측값:")
        for j_idx, (row, col, _) in enumerate(junction_indices):
            true_value = y_true[row, col]
            pred_value = y_pred[row, col]
            print(f"Junction {j_idx+1} (위치: [{row}, {col}]) -> True: {true_value:.4f}, Predicted: {pred_value:.4f}")

# 평가 함수 실행
def evaluate(flooding_file):
    model_path = '/content/ConvLSTM2D/model/convlstm_model.keras'
    model = load_trained_model(model_path)
    X_test, y_test = load_test_data()
    junction_mask = np.load("/content/ConvLSTM2D/DATA_numpy/junction_mask.npy")
    predictions, y_trues = evaluate_model(model, X_test, y_test, junction_mask)
    plot_and_save_prediction(y_trues, predictions)
    _, junction_mask, junction_indices = load_shapefile("/content/ConvLSTM2D/DATA_input/DEM/DEM_GRID.shp")
    inspect_predictions(y_trues, predictions, junction_indices, flooding_file)

if __name__ == "__main__":
    flooding_file = "/content/ConvLSTM2D/DATA_goal/Junction_Flooding_example.xlsx"
    evaluate(flooding_file)
