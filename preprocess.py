import geopandas as gpd
import numpy as np
import pandas as pd
import os
from datetime import datetime

# 파일 경로 설정
shapefile_path = '/content/ConvLSTM2D/DATA_input/DEM/DEM_GRID.shp'
rainfall_folder = '/content/ConvLSTM2D/DATA_input/RAINFALL'
flooding_folder = '/content/ConvLSTM2D/DATA_goal'
output_folder = '/content/ConvLSTM2D/DATA_numpy'

# Shapefile 데이터 로드 및 Junction 위치 매핑
def load_shapefile(shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    grid_size = (64, 64)
    data_array = np.zeros(grid_size, dtype=[('N_Imperv', 'f8'), ('N_perv', 'f8'), ('%Imperv', 'f8'),
                                            ('Elevation', 'f8'), ('Junction', 'U10'), ('left', 'f8'), 
                                            ('right', 'f8'), ('top', 'f8'), ('bottom', 'f8')])
    junction_locations = {}

    for _, row in gdf.iterrows():
        row_index, col_index = int(row['row_index']), int(row['col_index'])
        N_Imperv, N_perv = row['N_Imperv'], row['N_perv']
        percent_Imperv, Elevation = row['%imperv'], row['Elevation']
        Junction = row['Junction'] if pd.notnull(row['Junction']) else None
        left, right, top, bottom = row['left'], row['right'], row['top'], row['bottom']

        if Junction:
            junction_locations[Junction] = (row_index, col_index)

        data_array[row_index, col_index] = (N_Imperv, N_perv, percent_Imperv, Elevation, 
                                            Junction if Junction else '', left, right, top, bottom)
    
    return data_array, junction_locations

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

def load_flooding_data(flooding_file_path, junction_locations, grid_size=(64, 64)):
    flooding_df = pd.read_excel(flooding_file_path)
    flooding_df['Time'] = pd.to_datetime(flooding_df['Time'])
    flooding_df.set_index('Time', inplace=True)
    
    y_list = []
    for time_step in flooding_df.index:
        flooding_row = flooding_df.loc[time_step]
        flooding_grid = np.zeros(grid_size)  # 초기화 후 Junction에만 할당
        for junction, (row_idx, col_idx) in junction_locations.items():
            if junction in flooding_row:
                flooding_grid[row_idx, col_idx] = flooding_row[junction]
        y_list.append(flooding_grid)
    
    return np.array(y_list)

def prepare_dataset(rainfall_data, flooding_data, junction_locations, time_steps=4):
    X, y, masks = [], [], []
    grid_size = (64, 64, 1)
    
    for i in range(time_steps, len(flooding_data) + time_steps):
        if len(rainfall_data) < i:
            continue

        past_rainfall = rainfall_data.iloc[i - time_steps:i].values.flatten()
        if len(past_rainfall) != time_steps:
            continue
        past_rainfall_grid = np.array([np.full(grid_size, value) for value in past_rainfall])
        X.append(np.stack(past_rainfall_grid))

        flooding_grid = np.zeros(grid_size)
        mask = np.zeros(grid_size)  # Junction 위치 마스크
        
        for junction, (row_idx, col_idx) in junction_locations.items():
            flooding_grid[row_idx, col_idx, 0] = flooding_data[i - time_steps][row_idx, col_idx]
            mask[row_idx, col_idx, 0] = 1  # Junction 위치에만 1 설정
        
        y.append(flooding_grid)
        masks.append(mask)
    
    return np.stack(X), np.array(y), np.array(masks)

def process_all_files():
    rainfall_files = sorted([f for f in os.listdir(rainfall_folder) if f.startswith('rainfall_event_') and f.endswith('.dat')])
    flooding_files = sorted([f for f in os.listdir(flooding_folder) if f.startswith('Junction_Flooding_') and f.endswith('.xlsx')])

    if len(rainfall_files) != len(flooding_files):
        print("Mismatch in rainfall and flooding files.")
        return

    terrain_data, junction_locations = load_shapefile(shapefile_path)
    np.save(os.path.join(output_folder, "terrain_data.npy"), terrain_data)

    for i, (rainfall_file, flooding_file) in enumerate(zip(rainfall_files, flooding_files), start=1):
        rainfall_file_path = os.path.join(rainfall_folder, rainfall_file)
        flooding_file_path = os.path.join(flooding_folder, flooding_file)

        rainfall_data = load_rainfall_data(rainfall_file_path)
        flooding_data = load_flooding_data(flooding_file_path, junction_locations)

        X, y, masks = prepare_dataset(rainfall_data, flooding_data, junction_locations)

        np.save(os.path.join(output_folder, f"rainfall_X_{i}.npy"), X)
        np.save(os.path.join(output_folder, f"flooding_y_{i}.npy"), y)
        np.save(os.path.join(output_folder, f"mask_{i}.npy"), masks)

if __name__ == "__main__":
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    process_all_files()
