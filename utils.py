import numpy as np
import os

output_folder = '/content/ConvLSTM2D/DATA_numpy'

def load_train_val_data():
    train_indices = range(1, 201)  # 학습 데이터
    val_indices = range(201, 251)  # 검증 데이터
    X_train, y_train = concatenate_npy_files(train_indices)
    X_val, y_val = concatenate_npy_files(val_indices)
    return X_train, y_train, X_val, y_val

def load_test_data():
    """테스트 데이터 로드 함수."""
    test_indices = range(251, 301)  # 테스트 데이터 인덱스 범위
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
            
            # rain과 flood 쌍이 존재하는지 확인
            if os.path.exists(rainfall_path) and os.path.exists(flooding_path):
                # 강수 데이터 불러오기 및 지형 데이터 결합
                rainfall_data = np.load(rainfall_path)  # (4, 64, 64, 1)
                combined_input = np.concatenate([rainfall_data, terrain_data], axis=-1)  # (4, 64, 64, 5)로 결합
                
                # 각각의 입력과 출력 데이터 추가
                X_list.append(combined_input)
                y_list.append(np.load(flooding_path))  # 출력 데이터 추가
                
                # 다음 m 값으로 이동
                m += 1
            else:
                # 일치하는 쌍이 없으면 루프 종료
                break
    
    # 최종적으로 numpy stack을 사용하여 데이터를 5D 형태로 반환
    return np.stack(X_list), np.stack(y_list)

