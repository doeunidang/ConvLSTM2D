import numpy as np
from tensorflow.keras.models import load_model
from utils import load_test_data, load_test_masks
import matplotlib.pyplot as plt

# Junction 위치 정보를 로드하는 함수
def load_junction_locations(terrain_data_path):
    terrain_data = np.load(terrain_data_path, allow_pickle=True).item()
    junction_locations = {}
    for i in range(terrain_data.shape[0]):
        for j in range(terrain_data.shape[1]):
            junction = terrain_data[i, j]['Junction']
            if junction:  # Junction 정보가 있는 경우만 저장
                junction_locations[junction] = (i, j)
    return junction_locations

# 모델 로드
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# 모델 평가 함수
def evaluate_model(model, X_test, y_test, mask):
    predictions = model.predict(X_test)
    masked_predictions = predictions * mask  # 마스크 적용
    return masked_predictions

# Junction별 실제 값과 예측 값을 표로 출력
def print_junction_flooding_values(y_true, y_pred, junction_locations, index=0):
    junction_data = []
    for junction, (i, j) in junction_locations.items():
        true_value = y_true[index, i, j, 0]
        predicted_value = y_pred[index, i, j, 0]
        junction_data.append({
            'Junction': junction,
            'True Flooding': round(true_value, 2),
            'Predicted Flooding': round(predicted_value, 2)
        })

    # DataFrame으로 출력
    import pandas as pd
    df = pd.DataFrame(junction_data)
    print(df)

# 샘플 이미지로 저장하는 함수
def save_plot_prediction(y_true, y_pred, index=0, file_name="prediction_output.png"):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(y_true[index].reshape(64, 64), cmap='Blues')
    plt.title('True Flooding Output')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(y_pred[index].reshape(64, 64), cmap='Blues')
    plt.title('Predicted Flooding Output')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

# 디버깅용: 10x10 셀 flooding 예측 출력
def debug_flooding_values(y_pred, start_row=0, start_col=0, size=10):
    print("Flooding Values in Top-Left 10x10 Grid (Sample Prediction):")
    for i in range(start_row, start_row + size):
        for j in range(start_col, start_col + size):
            print(f"({i},{j}): {round(y_pred[0, i, j, 0], 2)}", end="  ")
        print()  # 다음 줄로 이동

# 평가 프로세스 함수
def evaluate():
    model_path = '/content/ConvLSTM2D/convlstm_model.keras'
    terrain_data_path = '/content/ConvLSTM2D/DATA_numpy/terrain_data.npy'
    
    # Junction 위치 정보 로드
    junction_locations = load_junction_locations(terrain_data_path)
    
    # 모델 및 테스트 데이터 로드
    model = load_trained_model(model_path)
    X_test, y_test = load_test_data()
    test_masks = load_test_masks()
    
    # 모델 평가 및 예측 수행
    predictions = evaluate_model(model, X_test, y_test, test_masks)
    
    # Junction별 예측 결과 출력
    print_junction_flooding_values(y_test, predictions, junction_locations, index=0)
    
    # 예측 결과 샘플 저장
    save_plot_prediction(y_test, predictions, index=0)
    
    # 10x10 셀 flooding 예측 디버깅 출력
    debug_flooding_values(predictions)

# 평가 실행
if __name__ == "__main__":
    evaluate()
