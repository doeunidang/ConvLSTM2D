import numpy as np
import os

# 데이터가 저장된 폴더 경로 설정
output_folder = '/content/ConvLSTM2D/DATA_numpy'

# 여러 개의 .npy 파일을 병합하는 함수
def concatenate_npy_files(indices, prefix):
    X_list = []  # X 데이터를 저장할 리스트
    y_list = []  # y 데이터를 저장할 리스트
    
    # 주어진 인덱스 리스트를 순회하며 .npy 파일을 로드하여 리스트에 추가
    for i in indices:
        X = np.load(os.path.join(output_folder, f'rainfall_X_{i}.npy'))
        y = np.load(os.path.join(output_folder, f'flooding_y_{i}.npy'))
        
        X_list.append(X)
        y_list.append(y)
    
    # 리스트에 저장된 X와 y 데이터를 축을 따라 병합
    X_concat = np.concatenate(X_list, axis=0)
    y_concat = np.concatenate(y_list, axis=0)
    
    return X_concat, y_concat

# 훈련 데이터와 검증 데이터를 로드하는 함수
def load_train_val_data():
    train_indices = range(1, 31)  # 훈련 데이터 파일 인덱스
    X_train, y_train = concatenate_npy_files(train_indices, 'rainfall')

    val_indices = range(31, 41)  # 검증 데이터 파일 인덱스
    X_val, y_val = concatenate_npy_files(val_indices, 'rainfall')
    
    return X_train, y_train, X_val, y_val

# 테스트 데이터를 로드하는 함수
def load_test_data():
    test_indices = range(41, 51)  # 테스트 데이터 파일 인덱스
    X_test, y_test = concatenate_npy_files(test_indices, 'rainfall')

    return X_test, y_test
