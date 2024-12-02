import numpy as np
import os

output_folder = '/content/ConvLSTM2D/DATA_numpy'

def load_train_val_data():
    train_indices = list(range(1, 81)) + list(range(101, 601)) + list(range(701, 781)) + list(range(901, 1001))  # 학습 데이터
    val_indices = list(range(81, 101)) + list(range(601, 701)) + list(range(781, 801)) + list(range(801, 901))  # 검증 데이터
    X_train, y_train = concatenate_npy_files(train_indices)
    X_val, y_val = concatenate_npy_files(val_indices)
    return X_train, y_train, X_val, y_val

def load_test_data():
    """테스트 데이터 로드 함수."""
    test_indices = range(1, 101)  # 테스트 데이터 인덱스 범위
    X_test, y_test = concatenate_npy_files(test_indices)
    return X_test, y_test

def concatenate_npy_files(indices):
    """rainfall와 flooding 파일 쌍을 일치시켜 데이터를 로드합니다."""
    X_list = []  # X 데이터를 저장할 리스트
    y_list = []  # y 데이터를 저장할 리스트
    terrain_path = os.path.join(output_folder, "terrain_data.npy")
    terrain_data = np.load(terrain_path, allow_pickle=True)  # (4, 64, 64, 4)

    for i in indices:
        m = 1
        while True:
            rainfall_path = os.path.join(output_folder, f'rainfall_{i}_{m}.npy')
            flooding_path = os.path.join(output_folder, f'flooding_{i}_{m}.npy')
            
            if os.path.exists(rainfall_path) and os.path.exists(flooding_path):
                rainfall_data = np.load(rainfall_path)
                combined_input = np.concatenate([rainfall_data, terrain_data], axis=-1)  # (4, 64, 64, 5)
                X_list.append(combined_input)
                y_list.append(np.load(flooding_path))
                m += 1
            else:
                break

    return np.stack(X_list), np.stack(y_list)
