import numpy as np
import os

# 경로 설정
output_folder = '/content/ConvLSTM2D/DATA_numpy'

# 여러 개의 .npy 파일을 하나로 병합하는 함수
def concatenate_npy_files(indices, prefix):
    X_list = []
    y_list = []
    mask_list = []
    
    # 각 인덱스에 대해 파일을 불러와 병합
    for i in indices:
        X = np.load(os.path.join(output_folder, f'rainfall_X_{i}.npy'))
        y = np.load(os.path.join(output_folder, f'flooding_y_{i}.npy'))
        mask = np.load(os.path.join(output_folder, f'mask_{i}.npy'))
        
        X_list.append(X)
        y_list.append(y)
        mask_list.append(mask)
    
    X_concat = np.concatenate(X_list, axis=0)
    y_concat = np.concatenate(y_list, axis=0)
    mask_concat = np.concatenate(mask_list, axis=0)
    
    return X_concat, y_concat, mask_concat

# 훈련 및 검증 데이터 로드 함수
def load_train_val_data():
    train_indices = range(1, 31)
    X_train, y_train, train_masks = concatenate_npy_files(train_indices, 'rainfall')

    val_indices = range(31, 41)
    X_val, y_val, val_masks = concatenate_npy_files(val_indices, 'rainfall')
    
    return X_train, y_train, X_val, y_val

def load_train_val_masks():
    train_indices = range(1, 31)
    _, _, train_masks = concatenate_npy_files(train_indices, 'rainfall')

    val_indices = range(31, 41)
    _, _, val_masks = concatenate_npy_files(val_indices, 'rainfall')
    
    return train_masks, val_masks
