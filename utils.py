import numpy as np
import os

# 경로 설정
output_folder = '/content/ConvLSTM2D/DATA_numpy'

# 여러 개의 .npy 파일을 하나로 병합하는 함수
def concatenate_npy_files(indices, prefix):
    X_list = []
    y_list = []
    
    # 각 인덱스에 대해 파일을 불러와 병합
    for i in indices:
        X = np.load(os.path.join(output_folder, f'rainfall_X_{i}.npy'))
        y = np.load(os.path.join(output_folder, f'flooding_y_{i}.npy'))
        
        X_list.append(X)
        y_list.append(y)
    
    # 파일들을 하나로 병합
    X_concat = np.concatenate(X_list, axis=0)
    y_concat = np.concatenate(y_list, axis=0)
    
    return X_concat, y_concat

# 훈련 데이터 로드 함수
def load_train_val_data():
    # 훈련 데이터 병합 및 로드
    train_indices = range(1, 31)  # 1~30번 파일 사용
    X_train, y_train = concatenate_npy_files(train_indices, 'rainfall')

    # 검증 데이터 병합 및 로드
    val_indices = range(31, 41)  # 31~40번 파일 사용
    X_val, y_val = concatenate_npy_files(val_indices, 'rainfall')
    
    return X_train, y_train, X_val, y_val

# 테스트 데이터 로드 함수
def load_test_data():
    # 테스트 데이터 병합 및 로드
    test_indices = range(41, 51)  # 41~50번 파일 사용
    X_test, y_test = concatenate_npy_files(test_indices, 'rainfall')

    return X_test, y_test
