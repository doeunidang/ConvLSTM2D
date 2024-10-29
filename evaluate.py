import numpy as np
from tensorflow.keras.models import load_model
from utils import load_test_data
import matplotlib.pyplot as plt

# 학습된 모델을 로드하는 함수
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# 테스트 데이터와 함께 모델 평가 함수
def evaluate_model(model, X_test, y_test):
    # 모델 성능 평가 (MSE 및 MAE 계산)
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss (MSE): {test_loss}")
    print(f"Test MAE: {test_mae}")
    
    # 테스트 데이터에 대한 예측
    predictions = model.predict(X_test)
    
    return predictions

# 예측 결과 시각화 함수
def plot_prediction(y_true, y_pred, index=0):
    plt.figure(figsize=(10, 5))

    # 실제 유출량 데이터
    plt.subplot(1, 2, 1)
    plt.imshow(y_true[index].reshape(64, 64), cmap='Blues')
    plt.title('True Flooding Output')
    plt.colorbar()

    # 예측된 유출량 데이터
    plt.subplot(1, 2, 2)
    plt.imshow(y_pred[index].reshape(64, 64), cmap='Blues')
    plt.title('Predicted Flooding Output')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# 평가 프로세스 함수
def evaluate():
    # 학습된 모델 경로 설정
    model_path = 'F:\\ConLSTM2D_TEST\\model\\convlstm_model.h5'
    
    # 모델 로드
    model = load_trained_model(model_path)
    
    # 테스트 데이터 로드
    X_test, y_test = load_test_data()
    
    # 모델 평가 및 예측 수행
    predictions = evaluate_model(model, X_test, y_test)
    
    # 예측 결과 시각화 (첫 번째 테스트 데이터)
    plot_prediction(y_test, predictions, index=0)

# 평가 실행
if __name__ == "__main__":
    evaluate()
