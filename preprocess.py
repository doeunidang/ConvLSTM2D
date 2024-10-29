import geopandas as gpd
import numpy as np
import pandas as pd
import os
from datetime import datetime

# 파일 경로 설정
shapefile_path = r'F:\\ConLSTM2D_TEST\\DATA_input\\DEM\\DEM_GRID.shp'
rainfall_folder = r'F:\\ConLSTM2D_TEST\\DATA_input\\RAINFALL'
flooding_folder = r'F:\\ConLSTM2D_TEST\\DATA_goal'
output_folder = r'F:\\ConLSTM2D_TEST\\DATA_numpy'

# 1. Shapefile 데이터 로드 및 Junction 위치 매핑
def load_shapefile(shapefile_path):
    gdf = gpd.read_file(shapefile_path)  # Shapefile을 GeoDataFrame으로 변환
    grid_size = (64, 64)  # 64x64 크기의 격자 설정
    
    # 각 셀에 다양한 속성을 저장할 배열 초기화
    data_array = np.zeros(grid_size, dtype=[('N_Imperv', 'f8'), ('N_perv', 'f8'), ('%Imperv', 'f8'),
                                            ('Elevation', 'f8'), ('Junction', 'U10'), ('left', 'f8'), 
                                            ('right', 'f8'), ('top', 'f8'), ('bottom', 'f8')])

    # Junction 위치를 저장할 딕셔너리
    junction_locations = {}

    # Shapefile의 각 행을 처리
    for _, row in gdf.iterrows():
        row_index = int(row['row_index'])
        col_index = int(row['col_index'])
        N_Imperv = row['N_Imperv']
        N_perv = row['N_perv']
        percent_Imperv = row['%imperv']
        Elevation = row['Elevation']
        Junction = row['Junction'] if pd.notnull(row['Junction']) else None
        left = row['left']
        right = row['right']
        top = row['top']
        bottom = row['bottom']

        # Junction이 있는 경우 Junction 위치를 기록
        if Junction:
            junction_locations[Junction] = (row_index, col_index)

        # 각 셀의 필드값을 data_array에 저장
        data_array[row_index, col_index] = (N_Imperv, N_perv, percent_Imperv, Elevation, 
                                            Junction if Junction else '', left, right, top, bottom)
    
    return data_array, junction_locations

# 2. 강우 데이터 로드 함수
def load_rainfall_data(rainfall_file_path, interval=10):
    data = []  # 강우 데이터를 저장할 리스트
    
    with open(rainfall_file_path, 'r') as file:
        for line in file:
            parts = line.split()
            # 날짜 및 강우량 정보를 추출
            year = int(parts[1])
            month = int(parts[2])
            day = int(parts[3])
            hour = int(parts[4])
            minute = int(parts[5])
            rainfall_value = float(parts[6])
            time = datetime(year, month, day, hour, minute)  # 시간 생성
            data.append([time, rainfall_value])
    
    # pandas DataFrame으로 변환
    rainfall_df = pd.DataFrame(data, columns=['Time', 'Rainfall'])
    rainfall_df.set_index('Time', inplace=True)
    
    return rainfall_df

# 3. 유출량 데이터 로드 함수 (Junction 정보를 64x64 격자에 매핑)
def load_flooding_data(flooding_file_path, junction_locations, grid_size=(64, 64)):
    flooding_df = pd.read_excel(flooding_file_path)
    flooding_df.index = pd.to_datetime(flooding_df['Time'])  # 시간 정보를 인덱스로 설정
    flooding_df.drop(columns='Time', inplace=True)  # Time 열을 제거
    
    y_list = []  # 유출량 데이터를 저장할 리스트
    
    for time_step in flooding_df.index:
        flooding_row = flooding_df.loc[time_step]
        flooding_grid = np.zeros(grid_size)  # 유출량 데이터를 저장할 격자 초기화
        
        for junction in flooding_row.index:
            if junction in junction_locations:
                row_idx, col_idx = junction_locations[junction]
                flooding_grid[row_idx, col_idx] = flooding_row[junction]  # 각 셀에 유출량 값 매핑
        
        y_list.append(flooding_grid)  # 시간별 유출량 데이터를 리스트에 추가
    
    y_array = np.array(y_list)  # 리스트를 numpy 배열로 변환
    return y_array

# 4. 데이터셋 준비 함수
def prepare_dataset(rainfall_data, flooding_data, time_steps=4):
    X, y = [], []  # 입력(X)과 출력(y) 데이터 저장 리스트

    # 각 타임스텝에 대해 데이터를 처리
    for i in range(time_steps, len(flooding_data) + time_steps):
        if len(rainfall_data) < i:
            print(f"Skipping step {i} as insufficient data for past {time_steps} steps.")
            continue
        
        # 이전 time_steps 동안의 강우 데이터를 추출
        past_rainfall = rainfall_data.iloc[i - time_steps:i].values.flatten()
        
        # 강우 데이터를 64x64로 변환
        if len(past_rainfall) != time_steps:
            print(f"Skipping step {i}: Insufficient past rainfall data.")
            continue

        past_rainfall_grid = np.array([np.full((64, 64, 1), value) for value in past_rainfall])
        if past_rainfall_grid.shape != (time_steps, 64, 64, 1):
            print(f"Error in grid shape at step {i}, got shape: {past_rainfall_grid.shape}")
        else:
            print(f"Grid shape at step {i} is correct: {past_rainfall_grid.shape}")
        
        X.append(np.stack(past_rainfall_grid))  # 입력 데이터로 추가

        # 타임스텝에 해당하는 유출량 데이터를 추출
        future_flooding = flooding_data[i - time_steps]
        future_flooding = future_flooding.reshape(64, 64, 1)
        y.append(future_flooding)  # 출력 데이터로 추가

    X = np.stack(X) if X else np.empty((0, time_steps, 64, 64, 1))
    y = np.array(y) if y else np.empty((0, 64, 64, 1))

    return X, y

# 5. 모든 파일에 대해 처리하고 결과를 저장하는 함수
def process_all_files():
    # 강우와 유출량 파일 목록을 검색
    rainfall_files = sorted([f for f in os.listdir(rainfall_folder) if f.startswith('rainfall_event_') and f.endswith('.dat')])
    flooding_files = sorted([f for f in os.listdir(flooding_folder) if f.startswith('Junction_Flooding_') and f.endswith('.xlsx')])

    # 파일 수가 일치하는지 확인
    if len(rainfall_files) != len(flooding_files):
        print("강우 데이터와 유출량 데이터의 파일 수가 일치하지 않습니다.")
        return

    # Shapefile 데이터 로드
    terrain_data, junction_locations = load_shapefile(shapefile_path)

    # 지형 데이터를 한 번만 저장 (모든 파일에 동일)
    np.save(os.path.join(output_folder, "terrain_data.npy"), terrain_data)
    print(f"Saved terrain data: shape {terrain_data.shape}")

    # 각 파일에 대해 데이터 처리
    for i, (rainfall_file, flooding_file) in enumerate(zip(rainfall_files, flooding_files), start=1):
        print(f"Processing files: {rainfall_file} and {flooding_file}")
        
        rainfall_file_path = os.path.join(rainfall_folder, rainfall_file)
        flooding_file_path = os.path.join(flooding_folder, flooding_file)

        # 강우량 데이터와 유출량 데이터 로드
        rainfall_data = load_rainfall_data(rainfall_file_path)
        flooding_data = load_flooding_data(flooding_file_path, junction_locations)

        # 데이터셋 준비
        X, y = prepare_dataset(rainfall_data, flooding_data)

        # 결과를 .npy 파일로 저장
        np.save(os.path.join(output_folder, f"rainfall_X_{i}.npy"), X)
        np.save(os.path.join(output_folder, f"flooding_y_{i}.npy"), y)
        print(f"Saved processed data for event {i}: X shape {X.shape}, y shape {y.shape}")

# 데이터 처리 실행
if __name__ == "__main__":
    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 모든 파일 처리 및 저장
    process_all_files()
