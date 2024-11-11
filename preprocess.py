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

def clear_output_folder(folder_path):
    """폴더 내 모든 파일 삭제"""
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

def sort_files_numerically(file_list):
    """숫자를 기준으로 파일 정렬"""
    return sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group()))

def load_shapefile(shapefile_path):
    """지형 데이터를 불러와 각 그리드 셀에 대한 특성 데이터를 저장하고, junction_mask를 생성합니다."""
    gdf = gpd.read_file(shapefile_path)
    grid_size = (64, 64)
    
    # 지형 정보를 저장할 데이터 배열과 junction 마스크 초기화
    data_array = np.zeros(grid_size, dtype=[('N_Imperv', 'f8'), ('N_perv', 'f8'), ('%Imperv', 'f8'), 
                                            ('Elevation', 'f8'), ('Junction', 'U10')])
    junction_mask = np.zeros(grid_size, dtype=bool)
    junction_indices = []

    # 지형 파일의 각 행을 순회하며 데이터 배열과 junction 마스크 생성
    for _, row in gdf.iterrows():
        row_index, col_index = int(row['row_index']), int(row['col_index'])
        
        data_array[row_index, col_index] = (
            row['N_Imperv'], row['N_perv'], row['%imperv'], row['Elevation'],
            row['Junction'] if pd.notnull(row['Junction']) else ''
        )
        
        # Junction이 존재하는 셀을 마스킹하고 인덱스 추가
        if pd.notnull(row['Junction']):
            junction_mask[row_index, col_index] = True
            junction_indices.append((row_index, col_index, row['Junction']))
    
    # Junction ID 기준으로 인덱스를 정렬하여 반환
    junction_indices.sort(key=lambda x: int(x[2][1:]))  # Junction 이름 기준으로 정렬
    
    # 지형 특성 채널을 개별로 확장하고 결합하여 (4, 64, 64, 4)로 만듦
    elevation_grid = data_array['Elevation'].reshape(64, 64, 1)
    imperv_grid = data_array['%Imperv'].reshape(64, 64, 1)
    n_imperv_grid = data_array['N_Imperv'].reshape(64, 64, 1)
    n_perv_grid = data_array['N_perv'].reshape(64, 64, 1)
    
    terrain_data = np.concatenate([elevation_grid, imperv_grid, n_imperv_grid, n_perv_grid], axis=-1)
    terrain_data = np.repeat(terrain_data[np.newaxis, ...], 4, axis=0)  # (4, 64, 64, 4) 형태로 확장

    # terrain_data와 junction_mask를 저장
    np.save(os.path.join(output_folder, "terrain_data.npy"), terrain_data)
    np.save(os.path.join(output_folder, "junction_mask.npy"), junction_mask)
    
    return terrain_data, junction_mask, junction_indices

def load_rainfall_data(rainfall_file_path):
    """강수 데이터를 불러와 시간별 강수량을 반환합니다."""
    data = []
    with open(rainfall_file_path, 'r') as file:
        for line in file:
            parts = line.split()
            year, month, day = int(parts[1]), int(parts[2]), int(parts[3])
            hour, minute, rainfall_value = int(parts[4]), int(parts[5]), float(parts[6])
            time = datetime(year, month, day, hour, minute)
            data.append([time, rainfall_value])
    return pd.DataFrame(data, columns=['Time', 'Rainfall']).set_index('Time')

def load_flooding_data(flooding_file_path, junction_indices, grid_size=(64, 64)):
    """홍수 데이터를 불러와 각 타임스텝에 대해 각 교차점 위치의 수위 값을 반환합니다."""
    flooding_df = pd.read_excel(flooding_file_path)
    flooding_df['Time'] = pd.to_datetime(flooding_df['Time'])
    flooding_df.set_index('Time', inplace=True)

    y_list = []
    for time_step in flooding_df.index:
        flooding_row = flooding_df.loc[time_step]
        flooding_grid = np.zeros(grid_size)
        
        # Junction 이름 순서를 매칭하여, Junction 위치에 맞는 유출량 값을 배치
        for (row_idx, col_idx, junction_name) in junction_indices:
            if junction_name in flooding_row.index:
                flooding_grid[row_idx, col_idx] = flooding_row[junction_name]
        
        y_list.append(flooding_grid.reshape(64, 64, 1))
    return np.array(y_list)

def save_time_window_data(data, prefix, index, folder):
    """타임 윈도우별 데이터를 저장"""
    np.save(os.path.join(folder, f"{prefix}_{index}.npy"), data)

def process_all_files():
    """모든 강수 및 홍수 파일을 처리하여 학습용 데이터셋을 생성합니다."""
    clear_output_folder(output_folder)

    # 파일 정렬
    rainfall_files = sort_files_numerically([f for f in os.listdir(rainfall_folder) if f.startswith('rainfall_event_') and f.endswith('.dat')])
    flooding_files = sort_files_numerically([f for f in os.listdir(flooding_folder) if f.startswith('Junction_Flooding_') and f.endswith('.xlsx')])

    # 지형 데이터 및 Junction 위치 로드
    terrain_data, junction_mask, junction_indices = load_shapefile(shapefile_path)

    for i, (rainfall_file, flooding_file) in enumerate(zip(rainfall_files, flooding_files), start=1):
        rainfall_data = load_rainfall_data(os.path.join(rainfall_folder, rainfall_file))
        flooding_data = load_flooding_data(os.path.join(flooding_folder, flooding_file), junction_indices)

        # 강수 데이터 타임스텝별 저장
        for j in range(len(rainfall_data) - 3):
            rainfall_segment = rainfall_data.iloc[j:j + 4]
            if len(rainfall_segment) == 4:
                past_rainfall_grid = np.array([np.full((64, 64, 1), value) for value in rainfall_segment['Rainfall'].values])
                save_time_window_data(past_rainfall_grid, f"rainfall_{i}", j + 1, output_folder)

        # 홍수 데이터 타임스텝별 저장
        for j in range(len(flooding_data) - 3):
            flooding_segment = flooding_data[j:j + 4]
            if len(flooding_segment) == 4:
                save_time_window_data(flooding_segment, f"flooding_{i}", j + 1, output_folder)

if __name__ == "__main__":
    process_all_files()  # 모든 파일 처리 함수 실행
