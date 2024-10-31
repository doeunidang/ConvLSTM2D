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

# 폴더가 없을 경우 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# shapefile을 로드하여 junction 위치를 매핑하는 함수
def load_shapefile(shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    grid_size = (64, 64)
    data_array = np.zeros(grid_size, dtype=[('N_Imperv', 'f8'), ('N_perv', 'f8'), ('%Imperv', 'f8'),
                                            ('Elevation', 'f8'), ('Junction', 'U10'), ('left', 'f8'), 
                                            ('right', 'f8'), ('top', 'f8'), ('bottom', 'f8')])
    junction_mask = np.zeros(grid_size, dtype=bool)
    junction_indices = []

    for _, row in gdf.iterrows():
        row_index, col_index = int(row['row_index']), int(row['col_index'])
        data_array[row_index, col_index] = (
            row['N_Imperv'], row['N_perv'], row['%imperv'], row['Elevation'], 
            row['Junction'] if pd.notnull(row['Junction']) else '', 
            row['left'], row['right'], row['top'], row['bottom']
        )
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
            year, month, day = int(parts[1]), int(parts[2]), int(parts[3])
            hour, minute, rainfall_value = int(parts[4]), int(parts[5]), float(parts[6])
            time = datetime(year, month, day, hour, minute)
            data.append([time, rainfall_value])
    rainfall_df = pd.DataFrame(data, columns=['Time', 'Rainfall']).set_index('Time')
    return rainfall_df

# 홍수 데이터를 junction 위치에 맞춰서 로드하는 함수
def load_flooding_data(flooding_file_path, junction_indices, grid_size=(64, 64)):
    flooding_df = pd.read_excel(flooding_file_path)
    flooding_df['Time'] = pd.to_datetime(flooding_df['Time'])
    flooding_df.set_index('Time', inplace=True)
    
    y_list = []
    
    for time_step in flooding_df.index:
        flooding_row = flooding_df.loc[time_step]
        flooding_values = np.zeros(len(junction_indices))
        
        for i, (row_idx, col_idx) in enumerate(junction_indices):
            junction = flooding_row.index[i] if i < len(flooding_row.index) else None
            flooding_values[i] = flooding_row[junction] if junction else 0
        
        y_list.append(flooding_values)
    
    y_array = np.array(y_list)
    return y_array

# 데이터셋을 준비하는 함수 (time_steps 길이의 시계열 강수량과 junction 위치에 따른 홍수량을 포함)
def prepare_dataset(rainfall_data, flooding_data, junction_indices, time_steps=4):
    X, y = [], []
    for i in range(time_steps, len(flooding_data)):
        if len(rainfall_data) < i:
            continue
        past_rainfall = rainfall_data.iloc[i - time_steps:i].values.flatten()
        
        past_rainfall_grid = np.array([np.full((64, 64, 1), value) for value in past_rainfall])
        X.append(np.stack(past_rainfall_grid))
        
        flooding_grid = np.zeros((64, 64))
        for (row_idx, col_idx), value in zip(junction_indices, flooding_data[i - time_steps]):
            flooding_grid[row_idx, col_idx] = value
        y.append(flooding_grid.reshape(64, 64, 1))

    return np.stack(X), np.array(y)

# 전체 파일을 처리하여 numpy 파일로 저장하는 함수
def process_all_files():
    rainfall_files = sorted([f for f in os.listdir(rainfall_folder) if f.startswith('rainfall_event_') and f.endswith('.dat')])
    flooding_files = sorted([f for f in os.listdir(flooding_folder) if f.startswith('Junction_Flooding_') and f.endswith('.xlsx')])

    if len(rainfall_files) != len(flooding_files):
        print("Mismatch in rainfall and flooding files.")
        return

    terrain_data, junction_mask, junction_indices = load_shapefile(shapefile_path)
    np.save(os.path.join(output_folder, "terrain_data.npy"), terrain_data)
    np.save(os.path.join(output_folder, "junction_mask.npy"), junction_mask)

    for i, (rainfall_file, flooding_file) in enumerate(zip(rainfall_files, flooding_files), start=1):
        rainfall_file_path = os.path.join(rainfall_folder, rainfall_file)
        flooding_file_path = os.path.join(flooding_folder, flooding_file)

        rainfall_data = load_rainfall_data(rainfall_file_path)
        flooding_data = load_flooding_data(flooding_file_path, junction_indices)

        X, y = prepare_dataset(rainfall_data, flooding_data, junction_indices)

        np.save(os.path.join(output_folder, f"rainfall_X_{i}.npy"), X)
        np.save(os.path.join(output_folder, f"flooding_y_{i}.npy"), y)

if __name__ == "__main__":
    process_all_files()
