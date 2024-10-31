import numpy as np
import os

output_folder = '/content/ConvLSTM2D/DATA_numpy'

# 여러 개의 .npy 파일을 병합하는 함수
def concatenate_npy_files(indices, prefix):
    X_list = []
    y_list = []
    
    for i in indices:
        X = np.load(os.path.join(output_folder, f'rainfall_X_{i}.npy'))
        y = np.load(os.path.join(output_folder, f'flooding_y_{i}.npy'))
        
        X_list.append(X)
        y_list.append(y)
    
    X_concat = np.concatenate(X_list, axis=0)
    y_concat = np.concatenate(y_list, axis=0)
    
    return X_concat, y_concat

# 훈련 데이터 로드 함수
def load_train_val_data():
    train_indices = range(1, 31)
    X_train, y_train = concatenate_npy_files(train_indices, 'rainfall')

    val_indices = range(31, 41)
    X_val, y_val = concatenate_npy_files(val_indices, 'rainfall')
    
    return X_train, y_train, X_val, y_val

# 테스트 데이터 로드 함수
def load_test_data():
    test_indices = range(41, 51)
    X_test, y_test = concatenate_npy_files(test_indices, 'rainfall')

    return X_test, y_test
