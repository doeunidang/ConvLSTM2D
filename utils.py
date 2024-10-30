# utils.py
import numpy as np
import os
import tensorflow as tf

output_folder = '/content/ConvLSTM2D/DATA_numpy'

# 여러 개의 .npy 파일을 하나로 병합하는 함수
def concatenate_npy_files(indices):
    X_list, y_list, masks_list = [], [], []
    for i in indices:
        X = np.load(os.path.join(output_folder, f'rainfall_X_{i}.npy'))
        y = np.load(os.path.join(output_folder, f'flooding_y_{i}.npy'))
        mask = np.load(os.path.join(output_folder, f'mask_{i}.npy'))

        X_list.append(X)
        y_list.append(y)
        masks_list.append(mask)

    X_concat = np.concatenate(X_list, axis=0)
    y_concat = np.concatenate(y_list, axis=0)
    masks_concat = np.concatenate(masks_list, axis=0)

    return X_concat, y_concat, masks_concat

# 훈련 및 검증 데이터 로드 함수
def load_train_val_data():
    train_indices = range(1, 31)
    X_train, y_train, train_masks = concatenate_npy_files(train_indices)

    val_indices = range(31, 41)
    X_val, y_val, val_masks = concatenate_npy_files(val_indices)

    return X_train, y_train, X_val, y_val, train_masks, val_masks

# 테스트 데이터 로드 함수
def load_test_data():
    test_indices = range(41, 51)
    X_test, y_test, test_masks = concatenate_npy_files(test_indices)

    return X_test, y_test, test_masks

# 마스크 확장 함수
def expand_mask_to_batch_size(mask, batch_size):
    mask_expanded = tf.expand_dims(mask, axis=0)
    mask_tiled = tf.tile(mask_expanded, [batch_size, 1, 1, 1])
    return mask_tiled
