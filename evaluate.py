import numpy as np
from tensorflow.keras.models import load_model
from utils import load_test_data
import matplotlib.pyplot as plt
import os
from preprocess import load_shapefile
from losses import custom_loss

# 저장된 모델을 로드하는 함수
def load_trained_model(model_path):
    model = load_model(model_path, custom_objects={"custom_loss": custom_loss})
    return model

# 모델을 평가하고 예측값과 실제값을 반환하는 함수
def evaluate_model(model, X_test, y_test, junction_mask):
    mask = np.logical_not(np.isnan(y_test[0]))  # NaN이 아닌 값만 사용하기 위한 마스크
    y_pred_series = model.predict(X_test)       # 모델로 예측 수행
    target_time_steps = [0, 1, 2, 3]            # 예측을 확인할 시간 스텝
    y_preds, y_trues = [], []                   # 예측값과 실제값을 저장할 리스트

    # 각 시간 스텝에 대한 예측과 실제값 추출
    for t in target_time_steps:
        y_pred_t = np.zeros_like(y_test[t])     # 예측 배열 초기화
        y_pred_t[mask] = y_pred_series[t][mask] # junction 위치에 대해서만 예측값 할당
        y_preds.append(y_pred_t)
        y_trues.append(y_test[t])

    return y_preds, y_trues

# 예측 결과를 시각화하고 파일로 저장하는 함수
def plot_and_save_prediction(y_trues, y_preds, output_folder="/content/ConvLSTM2D/results"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # 결과 저장 폴더 생성
    
    for idx, (y_true, y_pred) in enumerate(zip(y_trues, y_preds)):
        minutes_ahead = (idx + 1) * 10  # 예측 시간 설정
        plt.figure(figsize=(10, 5))
        
        # 실제 홍수 출력을 시각화
        plt.subplot(1, 2, 1)
        plt.imshow(y_true.reshape(64, 64), cmap='Blues')
        plt.title(f'True Flooding Output ({minutes_ahead} Minutes Ahead)')
        plt.colorbar()
        
        # 예측 홍수 출력을 시각화
        plt.subplot(1, 2, 2)
        plt.imshow(y_pred.reshape(64, 64), cmap='Blues')
        plt.title(f'Predicted Flooding Output ({minutes_ahead} Minutes Ahead)')
        plt.colorbar()
        
        output_path = os.path.join(output_folder, f'prediction_{minutes_ahead}min.png')
        plt.savefig(output_path)  # 예측 결과 이미지 저장
        plt.close()

        print(f"Prediction image for {minutes_ahead} minutes ahead saved at {output_path}")

# 특정 영역 및 junction 위치에 대해 예측값을 점검하는 함수
def inspect_predictions(y_trues, y_preds, junction_indices, flooding_file, region_size=30):
    print(f"\nTrue값으로 사용된 flooding 파일 이름: {flooding_file}")

    for idx, (y_true, y_pred) in enumerate(zip(y_trues, y_preds)):
        minutes_ahead = (idx + 1) * 10
        print(f"\nTime Step ({minutes_ahead} Minutes Ahead)")

        if y_true.ndim == 3:
            y_true = y_true[:, :, 0]
        if y_pred.ndim == 3:
            y_pred = y_pred[:, :, 0]

        # 지정된 30x30 영역 내 예측값 출력
        print("\n30x30 영역의 예측값:")
        for i in range(region_size):
            for j in range(region_size):
                print(f"{y_pred[i, j]:.4f}", end=" ")
            print()

        # 각 junction 위치에서의 true값과 예측값 비교 출력
        print("\n각 junction 위치의 true값과 예측값:")
        for j_idx, (row, col) in enumerate(junction_indices):
            true_value = y_true[row, col]
            pred_value = y_pred[row, col]
            print(f"Junction {j_idx+1} (위치: [{row}, {col}]) -> True: {true_value:.4f}, Predicted: {pred_value:.4f}")

# 모델 평가를 위한 주요 함수
def evaluate(flooding_file):
    model_path = '/content/ConvLSTM2D/model/convlstm_model.keras'
    model = load_trained_model(model_path)  # 학습된 모델 로드
    X_test, y_test = load_test_data()       # 테스트 데이터 로드
    
    junction_mask = np.load("/content/ConvLSTM2D/DATA_numpy/junction_mask.npy")
    predictions, y_trues = evaluate_model(model, X_test, y_test, junction_mask)
    
    plot_and_save_prediction(y_trues, predictions)  # 예측 결과 시각화 및 저장
    
    _, junction_mask, junction_indices = load_shapefile("/content/ConvLSTM2D/DATA_input/DEM/DEM_GRID.shp")
    
    inspect_predictions(y_trues, predictions, junction_indices, flooding_file)  # 예측값 점검

# 메인 실행 코드
if __name__ == "__main__":
    flooding_file = "/content/ConvLSTM2D/DATA_goal/Junction_Flooding_example.xlsx"
    evaluate(flooding_file)
