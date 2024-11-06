import geopandas as gpd
import numpy as np
import pandas as pd
import os
import re
from datetime import datetime

# 경로 설정
shapefile_path = '/content/ConvLSTM2D/DATA_input/DEM/DEM_GRID.shp'
rainfall_folder = '/content/ConvLSTM2D/DATA_input/RAINFALL'
flooding_folder = '/content/ConvLSTM2D/DATA_goal'
output_folder = '/content/ConvLSTM2D/DATA_numpy'

# 데이터 저장 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def sort_files_numerically(file_list):
    """파일 이름에서 숫자를 추출해 정렬합니다."""
    # 정규식을 이용하여 파일 이름의 숫자를 추출하고 정렬
    return sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group()))

def load_shapefile(shapefile_path):
    """지형 데이터를 불러와 각 그리드 셀에 대한 특성 데이터를 저장합니다."""
    gdf = gpd.read_file(shapefile_path)  # shapefile 데이터를 geopandas를 사용해 불러옴
    grid_size = (64, 64)  # 그리드 사이즈 설정
    # 지형 정보를 저장할 데이터 배열 생성
    data_array = np.zeros(grid_size, dtype=[('N_Imperv', 'f8'), ('N_perv', 'f8'), ('%Imperv', 'f8'), 
                                            ('Elevation', 'f8'), ('Junction', 'U10')])
    # 교차점 위치 마스크와 교차점 인덱스 리스트 초기화
    junction_mask = np.zeros(grid_size, dtype=bool)
    junction_indices = []

    # 지형 파일의 각 행을 순회하며 데이터 배열에 각 그리드의 특성 값을 할당
    for _, row in gdf.iterrows():
        row_index, col_index = int(row['row_index']), int(row['col_index'])
        # 각 특성 값을 배열에 할당
        data_array[row_index, col_index] = (
            row['N_Imperv'], row['N_perv'], row['%imperv'], row['Elevation'],
            row['Junction'] if pd.notnull(row['Junction']) else ''
        )
        # 교차점이 존재하는 셀을 마스킹하고 인덱스를 저장
        if pd.notnull(row['Junction']):
            junction_mask[row_index, col_index] = True
            junction_indices.append((row_index, col_index, row['Junction']))
    
    # Junction ID 기준으로 인덱스를 정렬하여 반환
    junction_indices.sort(key=lambda x: int(x[2][1:]))
    return data_array, junction_mask, junction_indices

def load_rainfall_data(rainfall_file_path, interval=10):
    """강수 데이터를 불러와 시간별 강수량을 반환합니다."""
    data = []  # 강수량 데이터를 저장할 리스트 초기화
    with open(rainfall_file_path, 'r') as file:
        # 각 줄을 읽어 날짜와 시간, 강수량을 저장
        for line in file:
            parts = line.split()
            year, month, day = int(parts[1]), int(parts[2]), int(parts[3])
            hour, minute, rainfall_value = int(parts[4]), int(parts[5]), float(parts[6])
            time = datetime(year, month, day, hour, minute)  # 날짜 및 시간 정보 생성
            data.append([time, rainfall_value])  # 시간과 강수량 값을 리스트에 추가
    # DataFrame으로 변환하고 시간 인덱스를 설정
    rainfall_df = pd.DataFrame(data, columns=['Time', 'Rainfall']).set_index('Time')
    return rainfall_df

def load_flooding_data(flooding_file_path, junction_indices, grid_size=(64, 64)):
    """홍수 데이터를 불러와 각 타임스텝에 대해 각 교차점 위치의 수위 값을 반환합니다."""
    flooding_df = pd.read_excel(flooding_file_path)  # 홍수 데이터를 엑셀 파일에서 읽어옴
    flooding_df['Time'] = pd.to_datetime(flooding_df['Time'])  # 시간 열을 datetime으로 변환
    flooding_df.set_index('Time', inplace=True)  # 시간 열을 인덱스로 설정
    
    y_list = []  # 각 시간 단계별 홍수 데이터를 저장할 리스트 초기화
    for time_step in flooding_df.index:
        flooding_row = flooding_df.loc[time_step]  # 현재 시간 스텝의 데이터 가져오기
        flooding_values = np.zeros(len(junction_indices))  # 교차점의 수위 값을 저장할 배열
        
        # 각 교차점 위치에 해당하는 수위 값을 할당
        for i, (row_idx, col_idx, _) in enumerate(junction_indices):
            junction = flooding_row.index[i] if i < len(flooding_row.index) else None
            flooding_values[i] = flooding_row[junction] if junction else 0
        y_list.append(flooding_values)  # 현재 시간 단계의 수위 값을 리스트에 추가
    
    y_array = np.array(y_list)  # 리스트를 배열로 변환하여 반환
    return y_array

def prepare_dataset(rainfall_data, flooding_data, junction_indices, terrain_path, time_steps=4):
    """과거 강우 데이터를 입력으로 하여 미래의 홍수 상황을 예측하기 위한 학습용 데이터셋을 준비합니다."""
    terrain_data = np.load(terrain_path, allow_pickle=True)  # 지형 데이터 불러오기
    # 지형 데이터의 각 특성을 개별적으로 3차원 배열로 생성
    elevation_grid = terrain_data['Elevation'].reshape(64, 64, 1)
    imperv_grid = terrain_data['%Imperv'].reshape(64, 64, 1)
    n_imperv_grid = terrain_data['N_Imperv'].reshape(64, 64, 1)
    n_perv_grid = terrain_data['N_perv'].reshape(64, 64, 1)

    # 지형 특성을 결합하여 4채널 특성 데이터 생성
    terrain_features = np.concatenate([elevation_grid, imperv_grid, n_imperv_grid, n_perv_grid], axis=-1)
    terrain_features = np.repeat(terrain_features[np.newaxis, ...], time_steps, axis=0)  # 타임스텝에 맞춰 확장

    X, y = [], []  # 입력(X)과 출력(y) 데이터를 저장할 리스트 초기화
    
    for i in range(len(rainfall_data) - 2 * time_steps):
        # 과거 강수 데이터를 타임스텝 길이만큼 슬라이싱
        past_rainfall = rainfall_data.iloc[i:i + time_steps].values.flatten()
        future_flooding = flooding_data[i + time_steps:i + 2 * time_steps]  # 미래 홍수 데이터 선택
        
        # 과거 강수 데이터를 그리드 형식으로 확장하여 결합
        past_rainfall_grid = np.array([np.full((64, 64, 1), value) for value in past_rainfall])
        combined_input = np.concatenate([past_rainfall_grid, terrain_features], axis=-1)
        X.append(combined_input)  # 결합된 입력 데이터를 X에 추가

        flooding_sequence = []  # 미래 홍수 데이터를 저장할 리스트
        # 각 타임스텝에 대해 그리드 형식으로 홍수 데이터를 할당
        for t in range(time_steps):
            flooding_grid = np.zeros((64, 64))  # 홍수 데이터가 없는 셀은 0으로 채우기
            for (row_idx, col_idx, _), value in zip(junction_indices, future_flooding[t]):
                flooding_grid[row_idx, col_idx] = value  # 교차점 위치에 홍수 데이터 값 할당
            flooding_sequence.append(flooding_grid.reshape(64, 64, 1))  # 형상을 맞추기 위해 채널 추가
        
        y.append(np.stack(flooding_sequence, axis=0))  # 타임스텝별 미래 홍수 데이터를 y에 추가

    return np.stack(X), np.array(y)  # 입력과 출력 데이터를 배열로 반환

def process_all_files():
    """모든 강수 및 홍수 파일을 처리하여 학습용 데이터셋을 생성합니다."""
    # 강수 및 홍수 파일을 정렬하여 순서대로 불러옴
    rainfall_files = sort_files_numerically([f for f in os.listdir(rainfall_folder) if f.startswith('rainfall_event_') and f.endswith('.dat')])
    flooding_files = sort_files_numerically([f for f in os.listdir(flooding_folder) if f.startswith('Junction_Flooding_') and f.endswith('.xlsx')])

    # 강수 및 홍수 파일 수가 맞지 않으면 오류 메시지 출력
    if len(rainfall_files) != len(flooding_files):
        print("Mismatch in rainfall and flooding files.")
        return

    # 지형 데이터 및 Junction 위치 마스크 저장
    terrain_path = os.path.join(output_folder, "terrain_data.npy")
    terrain_data, junction_mask, junction_indices = load_shapefile(shapefile_path)
    np.save(terrain_path, terrain_data)  # 지형 데이터 저장
    junction_mask_path = os.path.join(output_folder, "junction_mask.npy")
    np.save(junction_mask_path, junction_mask)  # Junction 마스크 저장

    # 각 강수 및 홍수 파일 쌍에 대해 데이터셋을 생성
    for i, (rainfall_file, flooding_file) in enumerate(zip(rainfall_files, flooding_files), start=1):
        rainfall_file_path = os.path.join(rainfall_folder, rainfall_file)
        flooding_file_path = os.path.join(flooding_folder, flooding_file)

        # 강수량과 홍수 데이터를 불러와 데이터셋 준비
        rainfall_data = load_rainfall_data(rainfall_file_path)
        flooding_data = load_flooding_data(flooding_file_path, junction_indices)
        
        # 준비된 X, y 데이터를 저장
        X, y = prepare_dataset(rainfall_data, flooding_data, junction_indices, terrain_path)
        np.save(os.path.join(output_folder, f"rainfall_X_{i}.npy"), X)
        np.save(os.path.join(output_folder, f"flooding_y_{i}.npy"), y)

if __name__ == "__main__":
    process_all_files()  # 모든 파일 처리 함수 실행
