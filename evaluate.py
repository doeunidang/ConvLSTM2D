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
    mask = np.logical_not(np.isnan(y_test[0]))
    y_pred_series = model.predict(X_test)
    
    y_preds, y_trues = [], []
    for t in range(4):  # 10분, 20분, 30분, 40분 후 예측을 위해
        y_pred_t = np.zeros_like(y_test[t])
        y_pred_t[mask] = y_pred_series[:, t][mask]
        y_preds.append(y_pred_t)
        y_trues.append(y_test[:, t])

    return y_preds, y_trues

def plot_and_save_prediction(y_trues, y_preds, output_folder="F:\\ConvLSTM2D_git\\ConvLSTM2D\\results"):
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

def inspect_predictions(y_trues, y_preds, junction_indices, actual_flooding_file, actual_rainfall_file):
    print(f"\n사용된 rainfall 파일 (입력): {actual_rainfall_file}")
    print(f"사용된 flooding 파일 (실제 값): {actual_flooding_file}\n")
    
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

def evaluate():
    model_path = 'F:\\ConvLSTM2D_git\\ConvLSTM2D\\model\\convlstm_model.keras'
    model = load_trained_model(model_path)
    X_test, y_test = load_test_data()
    junction_mask = np.load("F:\\ConvLSTM2D_git\\ConvLSTM2D/DATA_numpy/junction_mask.npy")
    predictions, y_trues = evaluate_model(model, X_test, y_test, junction_mask)
    
    rainfall_file = 'F:\\ConvLSTM2D_git\\ConvLSTM2D\\DATA_input\\RAINFALL\\rainfall_event_251.dat'
    flooding_file = 'F:\\ConvLSTM2D_git\\ConvLSTM2D\\DATA_goal\\Junction_Flooding_251.xlsx'

    plot_and_save_prediction(y_trues, predictions)
    _, junction_mask, junction_indices = load_shapefile("F:\\ConvLSTM2D_git\\ConvLSTM2D\\DATA_input\\DEM\\DEM_GRID.shp")
    inspect_predictions(y_trues, predictions, junction_indices, flooding_file, rainfall_file)

if __name__ == "__main__":
    evaluate()
