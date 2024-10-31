import numpy as np
import os

output_folder = '/content/ConvLSTM2D/DATA_numpy'

# 데이터 병합 함수
def concatenate_npy_files(indices, prefix):
    X_list, y_list = [], []
    for i in indices:
        X = np.load(os.path.join(output_folder, f'rainfall_X_{i}.npy'))
        y = np.load(os.path.join(output_folder, f'flooding_y_{i}.npy'))
        X_list.append(X)
        y_list.append(y)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

# 학습 및 검증 데이터 로드
def load_train_val_data():
    train_indices = range(1, 201)  # 학습 데이터
    X_train, y_train = concatenate_npy_files(train_indices, 'rainfall')
    val_indices = range(201, 251)  # 검증 데이터
    X_val, y_val = concatenate_npy_files(val_indices, 'rainfall')
    return X_train, y_train, X_val, y_val

# 테스트 데이터 로드
def load_test_data():
    test_indices = range(251, 301)  # 테스트 데이터
    X_test, y_test = concatenate_npy_files(test_indices, 'rainfall')
    return X_test, y_test
