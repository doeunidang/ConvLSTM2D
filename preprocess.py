import geopandas as gpd
import numpy as np
import pandas as pd
import os
from datetime import datetime

# 경로 설정
shapefile_path = '/content/ConvLSTM2D/DATA_input/DEM/DEM_GRID.shp'
rainfall_folder = '/content/ConvLSTM2D/DATA_input/RAINFALL'
flooding_folder = '/content/ConvLSTM2D/DATA_goal'
output_folder = '/content/ConvLSTM2D/DATA_numpy'

# 출력 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# shapefile을 로드하고 junction 위치를 grid 데이터에 매핑하는 함수
def load_shapefile(shapefile_path):
    gdf = gpd.read_file(shapefile_path)  # shapefile 로드
    grid_size = (64, 64)  # grid 크기 설정
    # grid 데이터를 저장할 배열 초기화 (여러 속성 포함)
    data_array = np.zeros(grid_size, dtype=[('N_Imperv', 'f8'), ('N_perv', 'f8'), ('%Imperv', 'f8'),
                                            ('Elevation', 'f8'), ('Junction', 'U10'), ('left', 'f8'), 
                                            ('right', 'f8'), ('top', 'f8'), ('bottom', 'f8')])
    junction_mask = np.zeros(grid_size, dtype=bool)  # junction 위치를 나타내는 마스크
    junction_indices = []  # junction 위치의 인덱스 저장

    # shapefile의 각 행을 순회하며 grid에 데이터 설정
    for _, row in gdf.iterrows():
        row_index, col_index = int(row['row_index']), int(row['col_index'])
        # 해당 위치에 속성 값 설정
        data_array[row_index, col_index] = (
            row['N_Imperv'], row['N_perv'], row['%imperv'], row['Elevation'], 
            row['Junction'] if pd.notnull(row['Junction']) else '', 
            row['left'], row['right'], row['top'], row['bottom']
        )
        # junction 위치 마스크와 인덱스 추가
        if pd.notnull(row['Junction']):
            junction_mask[row_index, col_index] = True
            junction_indices.append((row_index, col_index))
    
    return data_array, junction_mask, junction_indices

# 강수량 데이터를 로드하는 함수
def load_rainfall_data(rainfall_file_path, interval=10):
    data = []
    with open(rainfall_file_path, 'r') as file:
        for line in file:
            parts = line.split()
            # 시간 정보 및 강수량 값 추출
            year, month, day = int(parts[1]), int(parts[2]), int(parts[3])
            hour, minute, rainfall_value = int(parts[4]), int(parts[5]), float(parts[6])
            time = datetime(year, month, day, hour, minute)
            data.append([time, rainfall_value])
    # 시간에 따라 인덱싱된 데이터프레임 생성
    rainfall_df = pd.DataFrame(data, columns=['Time', 'Rainfall']).set_index('Time')
    return rainfall_df

# 홍수 데이터를 junction 위치에 맞춰서 로드하는 함수
def load_flooding_data(flooding_file_path, junction_indices, grid_size=(64, 64)):
    flooding_df = pd.read_excel(flooding_file_path)  # 홍수 데이터 로드
    flooding_df['Time'] = pd.to_datetime(flooding_df['Time'])
    flooding_df.set_index('Time', inplace=True)
    
    y_list = []  # 시간별 홍수 데이터를 저장할 리스트
    
    for time_step in flooding_df.index:
        flooding_row = flooding_df.loc[time_step]
        flooding_values = np.zeros(len(junction_indices))
        
        # junction 위치의 데이터를 시간 순서대로 배열에 저장
        for i, (row_idx, col_idx) in enumerate(junction_indices):
            junction = flooding_row.index[i] if i < len(flooding_row.index) else None
            flooding_values[i] = flooding_row[junction] if junction else 0
        
        y_list.append(flooding_values)
    
    y_array = np.array(y_list)
    return y_array

# 데이터셋을 준비하는 함수 (강수량 및 홍수 데이터를 기반으로)
def prepare_dataset(rainfall_data, flooding_data, junction_indices, time_steps=4):
    X, y = [], []
    for i in range(time_steps, len(flooding_data)):
        if len(rainfall_data) < i:
            continue
        # 과거 강수량 데이터를 지정된 시간 단계로 슬라이싱
        past_rainfall = rainfall_data.iloc[i - time_steps:i].values.flatten()
        
        # 강수량 데이터를 64x64 형태의 그리드로 변환
        past_rainfall_grid = np.array([np.full((64, 64, 1), value) for value in past_rainfall])
        X.append(np.stack(past_rainfall_grid))
        
        # junction 위치에만 홍수 데이터 설정 (나머지는 NaN)
        flooding_grid = np.full((64, 64), np.nan)
        for (row_idx, col_idx), value in zip(junction_indices, flooding_data[i - time_steps]):
            flooding_grid[row_idx, col_idx] = value  # junction 위치에만 값을 설정
        y.append(flooding_grid.reshape(64, 64, 1))

    return np.stack(X), np.array(y)

# 전체 파일을 처리하여 numpy 파일로 저장하는 함수
def process_all_files():
    # 강수량 및 홍수 데이터 파일 목록 불러오기
    rainfall_files = sorted([f for f in os.listdir(rainfall_folder) if f.startswith('rainfall_event_') and f.endswith('.dat')])
    flooding_files = sorted([f for f in os.listdir(flooding_folder) if f.startswith('Junction_Flooding_') and f.endswith('.xlsx')])

    # 강수량과 홍수 파일 개수가 일치하는지 확인
    if len(rainfall_files) != len(flooding_files):
        print("강수량과 홍수 파일 수가 일치하지 않습니다.")
        return

    # 지형 데이터 및 junction 정보를 로드하여 저장
    terrain_data, junction_mask, junction_indices = load_shapefile(shapefile_path)
    np.save(os.path.join(output_folder, "terrain_data.npy"), terrain_data)
    np.save(os.path.join(output_folder, "junction_mask.npy"), junction_mask)

    # 각 강수량 및 홍수 파일 쌍에 대해 데이터 처리 및 저장
    for i, (rainfall_file, flooding_file) in enumerate(zip(rainfall_files, flooding_files), start=1):
        rainfall_file_path = os.path.join(rainfall_folder, rainfall_file)
        flooding_file_path = os.path.join(flooding_folder, flooding_file)

        rainfall_data = load_rainfall_data(rainfall_file_path)
        flooding_data = load_flooding_data(flooding_file_path, junction_indices)

        X, y = prepare_dataset(rainfall_data, flooding_data, junction_indices)

        # numpy 형식으로 강수량(X) 및 홍수 데이터(y) 저장
        np.save(os.path.join(output_folder, f"rainfall_X_{i}.npy"), X)
        np.save(os.path.join(output_folder, f"flooding_y_{i}.npy"), y)

if __name__ == "__main__":
    process_all_files()
