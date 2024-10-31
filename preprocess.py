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
            junction_indices.append((row_index, col_index, row['Junction']))
    
    junction_indices.sort(key=lambda x: int(x[2][1:]))  # Junction ID 순서대로 정렬
    return data_array, junction_mask, junction_indices

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

def load_flooding_data(flooding_file_path, junction_indices, grid_size=(64, 64)):
    flooding_df = pd.read_excel(flooding_file_path)
    flooding_df['Time'] = pd.to_datetime(flooding_df['Time'])
    flooding_df.set_index('Time', inplace=True)
    
    y_list = []
    for time_step in flooding_df.index:
        flooding_row = flooding_df.loc[time_step]
        flooding_values = np.zeros(len(junction_indices))
        
        for i, (row_idx, col_idx, _) in enumerate(junction_indices):
            junction = flooding_row.index[i] if i < len(flooding_row.index) else None
            flooding_values[i] = flooding_row[junction] if junction else 0
        y_list.append(flooding_values)
    
    y_array = np.array(y_list)
    return y_array

def prepare_dataset(rainfall_data, flooding_data, junction_indices, terrain_path, time_steps=4):
    terrain_data = np.load(terrain_path, allow_pickle=True)
    elevation_grid = terrain_data['Elevation'].reshape(64, 64, 1)
    imperv_grid = terrain_data['%Imperv'].reshape(64, 64, 1)
    n_imperv_grid = terrain_data['N_Imperv'].reshape(64, 64, 1)
    n_perv_grid = terrain_data['N_perv'].reshape(64, 64, 1)
    left_grid = terrain_data['left'].reshape(64, 64, 1)
    right_grid = terrain_data['right'].reshape(64, 64, 1)
    top_grid = terrain_data['top'].reshape(64, 64, 1)
    bottom_grid = terrain_data['bottom'].reshape(64, 64, 1)
    
    terrain_features = np.concatenate(
        [elevation_grid, imperv_grid, n_imperv_grid, n_perv_grid, left_grid, right_grid, top_grid, bottom_grid], axis=-1
    )
    terrain_features = np.repeat(terrain_features[np.newaxis, ...], time_steps, axis=0) 

    X, y = [], []
    for i in range(time_steps, len(flooding_data)):
        if len(rainfall_data) < i:
            continue
        
        past_rainfall = rainfall_data.iloc[i - time_steps:i].values.flatten()
        past_rainfall_grid = np.array([np.full((64, 64, 1), value) for value in past_rainfall])
        combined_input = np.concatenate([past_rainfall_grid, terrain_features], axis=-1)
        X.append(combined_input)
        
        flooding_grid = np.full((64, 64), np.nan)
        for (row_idx, col_idx, _), value in zip(junction_indices, flooding_data[i - time_steps]):
            flooding_grid[row_idx, col_idx] = value
        y.append(flooding_grid.reshape(64, 64, 1))

    return np.stack(X), np.array(y)

def process_all_files():
    rainfall_files = sorted([f for f in os.listdir(rainfall_folder) if f.startswith('rainfall_event_') and f.endswith('.dat')])
    flooding_files = sorted([f for f in os.listdir(flooding_folder) if f.startswith('Junction_Flooding_') and f.endswith('.xlsx')])

    if len(rainfall_files) != len(flooding_files):
        print("Mismatch in rainfall and flooding files.")
        return

    terrain_path = os.path.join(output_folder, "terrain_data.npy")
    terrain_data, junction_mask, junction_indices = load_shapefile(shapefile_path)
    np.save(terrain_path, terrain_data)
    
    # junction_mask 파일 저장
    junction_mask_path = os.path.join(output_folder, "junction_mask.npy")
    np.save(junction_mask_path, junction_mask)

    for i, (rainfall_file, flooding_file) in enumerate(zip(rainfall_files, flooding_files), start=1):
        rainfall_file_path = os.path.join(rainfall_folder, rainfall_file)
        flooding_file_path = os.path.join(flooding_folder, flooding_file)

        rainfall_data = load_rainfall_data(rainfall_file_path)
        flooding_data = load_flooding_data(flooding_file_path, junction_indices)

        X, y = prepare_dataset(rainfall_data, flooding_data, junction_indices, terrain_path)

        np.save(os.path.join(output_folder, f"rainfall_X_{i}.npy"), X)
        np.save(os.path.join(output_folder, f"flooding_y_{i}.npy"), y)

if __name__ == "__main__":
    process_all_files()
