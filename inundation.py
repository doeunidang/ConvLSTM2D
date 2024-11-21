import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from collections import deque
from tensorflow.keras.models import load_model
import os
from utils import load_test_data
from preprocess import load_shapefile
from losses import custom_loss

# 경로 설정
model_path = '/content/ConvLSTM2D/model/convlstm_model.keras'
shapefile_path = '/content/ConvLSTM2D/DATA_input/DEM/DEM_GRID.shp'
output_folder = '/content/ConvLSTM2D/results'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 모델 예측 함수 (단일 샘플)
def predict_discharge(sample_idx=0):
    model = load_model(model_path, custom_objects={'custom_loss': custom_loss})
    X_test, y_test = load_test_data()  # 전체 테스트 데이터 로드
    X_test_sample = X_test[sample_idx:sample_idx+1]  # 단일 샘플 선택
    y_test_sample = y_test[sample_idx:sample_idx+1]
    y_pred_sample = model.predict(X_test_sample)  # 단일 샘플 예측 수행
    return X_test_sample, y_test_sample, y_pred_sample

# Junction별로 예측 결과를 출력
def print_junction_discharge(y_pred_sample, junction_indices):
    time_steps = ["t-20", "t-10", "t", "t+10"]
    print("모델 예측 결과 (Junction별 유출량):")
    for row, col, junction_name in junction_indices:
        print(f"\n{junction_name} (위치: [{row}, {col}])")
        for t, time_label in enumerate(time_steps):
            pred_value = y_pred_sample[0, t, row, col, 0]  # 예측된 유출량 값
            print(f"{time_label} - Predicted: {pred_value:.4f}")

# 지형 데이터 로드 및 Junction 데이터 초기화
def load_shapefile_and_initialize_grid(shapefile_path, y_pred_sample):
    gdf = gpd.read_file(shapefile_path)
    grid_data = gdf[['row_index', 'col_index', 'Elevation', 'Junction']].copy()
    
    grid_data['Junction'] = grid_data['Junction'].apply(lambda x: f"J{int(x)}" if pd.notnull(x) and not str(x).startswith('J') else x)
    
    # Junction별 예측값을 grid_data에 통합
    for t in range(4):
        grid_data.loc[:, f'flooding_value_{t+1}'] = np.nan  # 예측값 컬럼 초기화
    
    for idx, row in grid_data.iterrows():
        junction_name = row['Junction']
        if pd.notnull(junction_name):
            j_idx = int(junction_name[1:]) - 1  # 'J1' => 0 인덱스 변환
            for t in range(4):
                row_index = int(row['row_index'])  # 정수형으로 변환
                col_index = int(row['col_index'])  # 정수형으로 변환
                grid_data.at[idx, f'flooding_value_{t+1}'] = y_pred_sample[t, row_index, col_index, 0]  # 4차원 인덱싱으로 수정
    
    return grid_data

# grid_data를 grid_array로 변환
def initialize_grid_array(grid_data):
    grid_array = np.zeros((64, 64), dtype=[('elevation', 'f8'), ('flooding_value', 'f8')])
    for _, row in grid_data.iterrows():
        x, y = int(row['col_index']), int(row['row_index'])
        grid_array[y, x] = (row['Elevation'], row[['flooding_value_1', 'flooding_value_2', 'flooding_value_3', 'flooding_value_4']].sum())
    return grid_array

# 침수 최저점 찾기 함수
def find_inundation_low_points(x, y, grid_array):
    queue = deque([(x, y)])
    visited = set()
    visited.add((x, y))
    lowest_points = [(x, y)]
    lowest_elevation = grid_array[y, x]['elevation']
    while queue:
        current_x, current_y = queue.popleft()
        current_elevation = grid_array[current_y, current_x]['elevation']
        neighbors = [(current_x + dx, current_y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx, dy) != (0, 0)]
        for nx, ny in neighbors:
            if 0 <= nx < 64 and 0 <= ny < 64 and (nx, ny) not in visited:
                visited.add((nx, ny))
                neighbor_elevation = grid_array[ny, nx]['elevation']
                if neighbor_elevation < lowest_elevation:
                    lowest_points = [(nx, ny)]
                    lowest_elevation = neighbor_elevation
                elif neighbor_elevation == lowest_elevation:
                    lowest_points.append((nx, ny))
                if neighbor_elevation <= current_elevation:
                    queue.append((nx, ny))
    return lowest_points, lowest_elevation

# 같은 고도의 셀 탐색 함수
def find_connected_same_elevation_cells(x, y, elevation, grid_array):
    queue = deque([(x, y)])
    visited = set()
    visited.add((x, y))
    connected_cells = [(x, y)]
    while queue:
        current_x, current_y = queue.popleft()
        neighbors = [(current_x + dx, current_y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx, dy) != (0, 0)]
        for nx, ny in neighbors:
            if 0 <= nx < 64 and 0 <= ny < 64 and (nx, ny) not in visited:
                if grid_array[ny, nx]['elevation'] == elevation:
                    queue.append((nx, ny))
                    visited.add((nx, ny))
                    connected_cells.append((nx, ny))
    return connected_cells

# 침수 깊이 계산 함수
def compute_total_flooding(H, elevation_groups, cell_area):
    total_flooding_computed = 0
    for elevation, cells in elevation_groups.items():
        flooded_cells_count = len(cells)
        total_flooding_computed += (H - elevation) * cell_area * flooded_cells_count
    return total_flooding_computed

# 침수 깊이 최적화 함수
def find_optimal_H(total_flooding, elevation_groups, cell_area, H_min, H_max, tolerance=1e-5):    
    while H_max - H_min > tolerance:
        H_mid = (H_min + H_max) / 2
        total_flooding_computed = compute_total_flooding(H_mid, elevation_groups, cell_area)
        
        if total_flooding_computed < total_flooding:
            H_min = H_mid
        else:
            H_max = H_mid
    return (H_min + H_max) / 2

def visualize_and_save_flooded_area(grid_array, flooded_cells, sample_idx, final_H, final_depths):
    # 플롯 생성
    plt.figure(figsize=(10, 10))
    elevation_array = grid_array['elevation'].copy()
    elevation_array[elevation_array == 999] = np.nan
    cmap = mpl.colormaps["terrain"].copy()  # Matplotlib 최신 버전에 맞게 수정
    cmap.set_bad(color='black')
    norm = plt.Normalize(vmin=-1, vmax=np.nanmax(elevation_array))
    plt.imshow(elevation_array, cmap=cmap, norm=norm, origin='lower')

    # 침수심 정규화
    min_inundation_H = float('inf')
    max_inundation_H = float('-inf')
    for cx, cy in flooded_cells:
        cell_elevation = grid_array[cy, cx]['elevation']
        inundation_H = final_H - cell_elevation  # final_H를 사용하도록 수정
        min_inundation_H = min(min_inundation_H, inundation_H)
        max_inundation_H = max(max_inundation_H, inundation_H)
    for cx, cy in flooded_cells:
        cell_elevation = grid_array[cy, cx]['elevation']
        inundation_H = final_H - cell_elevation  # final_H를 사용하도록 수정
        normalized_inundation_H = (inundation_H - min_inundation_H) / (max_inundation_H - min_inundation_H)

        # 침수심별 색상 매핑
        if normalized_inundation_H <= 0.2:
            color = 'cornflowerblue'
        elif 0.2 < normalized_inundation_H <= 0.3:
            color = 'royalblue'
        elif 0.3 < normalized_inundation_H <= 0.5:
            color = 'mediumblue'
        else:
            color = 'darkblue'
        plt.plot(cx, cy, 's', markersize=10, color=color)

    # Y축 반전 및 제목/범례 추가
    plt.gca().invert_yaxis()
    plt.title(f'Flooded Areas and Elevation Map - Sample {sample_idx}\nFlood Depth (H): {final_H:.2f}')
    legend_elements = [
        Patch(color='cornflowerblue', label='0 ~ 0.2m'),
        Patch(color='royalblue', label='0.2 ~ 0.3m'),
        Patch(color='mediumblue', label='0.3 ~ 0.5m'),
        Patch(color='darkblue', label='0.5m ~')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # 저장 경로 설정 및 이미지 저장
    image_path = os.path.join(output_folder, f'flooded_area_sample_{sample_idx}.png')
    plt.savefig(image_path, dpi=300)  # 해상도를 300 DPI로 설정
    plt.close()
    print(f"Saved flooded area visualization for sample {sample_idx} at {image_path}")

    # 고도 그룹별 침수 심도 출력 (침수된 셀들만, 음수 제외)
    for elevation, depths in final_depths.items():
        if depths:  # depths가 비어있지 않으면
            flood_depth = depths[0]
            if flood_depth >= 0:  # 음수 침수 심도는 제외
                print(f"Elevation: {elevation}, Flood Depth: {flood_depth}")


# 메인 함수
def main(sample_idx=0):
    X_test_sample, y_test_sample, y_pred_sample = predict_discharge(sample_idx)
    
    # 1. Junction별 예측 결과 출력
    junction_indices = load_shapefile(shapefile_path)[2]  # junction_indices는 load_shapefile의 반환 값 중 세 번째 요소
    print_junction_discharge(y_pred_sample, junction_indices)
    
    grid_data = load_shapefile_and_initialize_grid(shapefile_path, y_pred_sample[0])
    grid_array = initialize_grid_array(grid_data)
    
    initial_flooded_cells = []
    for _, row in grid_data.iterrows():
        junction_id = row['Junction']
        # t+10 값만 사용하도록 수정
        flooding_value = row['flooding_value_4']  # t+10 값을 가져옵니다.
        
        # Junction ID 위치 찾기
        flood_cell = grid_data[grid_data['Junction'] == junction_id]
        if not flood_cell.empty:
            x, y = int(flood_cell.iloc[0]['col_index']), int(flood_cell.iloc[0]['row_index'])
        
            # 침수 최저점 찾기
            low_points, elevation = find_inundation_low_points(x, y, grid_array)
        
            # 초기 침수 범위 설정
            for low_x, low_y in low_points:
                initial_flooded_cells.extend(find_connected_same_elevation_cells(low_x, low_y, elevation, grid_array))
    
    flooded_cells = set(initial_flooded_cells)
    
    # np.min 사용하여 최저 고도 값 구하기
    H_min = np.min(grid_array['elevation'])  # 최저 고도
    H_max = 41.68772125  # 최대 고도
    
    # 고도별로 침수 셀을 그룹화하여 elevation_groups 생성
    elevation_groups = {}
    for x, y in flooded_cells:
        cell_elevation = grid_array[y, x]['elevation']
        if cell_elevation != 999:
            if cell_elevation not in elevation_groups:
                elevation_groups[cell_elevation] = []
            elevation_groups[cell_elevation].append((x, y))

    # t+10의 값만 총유출량으로 사용
    total_flooding = y_pred_sample[0, 3, :, :, 0].sum() * 600  # t+10의 값만 사용하여 총유출량 계산
    cell_area = 244.1406
    final_H = find_optimal_H(total_flooding, elevation_groups, cell_area, H_min, H_max)

    # 침수 범위 확장 (H 값에 맞는 고도 영역까지 확장)
    while True:
        new_flooded_cells = set(flooded_cells)  # 현재 flooded_cells 복사본을 생성
        max_depth = final_H  # 최대 수심을 최적화된 H 값으로 설정

        # 침수된 셀의 모든 인접 셀의 고도를 수집하기 위해 사용될 집합
        all_higher_adjacent_elevations = set()

        # 인접 셀을 순회하여 침수 범위를 확장
        for x, y in flooded_cells:
            neighbors = [(x + dx, y + dy) 
                         for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                         if (dx != 0 or dy != 0)]
            
            for nx, ny in neighbors:
                if 0 <= nx < 64 and 0 <= ny < 64:
                    if (nx, ny) not in flooded_cells:  # 이미 침수된 영역은 탐색에서 제외
                        adjacent_elevation = grid_array[ny, nx]['elevation']
                        if adjacent_elevation <= max_depth and adjacent_elevation != 999:  # 최대 수심 이하의 셀을 확장
                            new_flooded_cells.add((nx, ny))

        if len(new_flooded_cells) == len(flooded_cells):  # 더 이상 확장되지 않으면 종료
            break
        
        flooded_cells = new_flooded_cells  # flooded_cells 업데이트

        # 고도별로 다시 그룹화
        elevation_groups = {}
        for x, y in flooded_cells:
            cell_elevation = grid_array[y, x]['elevation']
            if cell_elevation != 999:
                if cell_elevation not in elevation_groups:
                    elevation_groups[cell_elevation] = []
                elevation_groups[cell_elevation].append((x, y))

        # H 값 최적화
        final_H = find_optimal_H(total_flooding, elevation_groups, cell_area, H_min, H_max)

    # 고도 그룹별 침수 깊이 계산
    final_depths = {}
    for x, y in flooded_cells:
        cell_elevation = grid_array[y, x]['elevation']
        if cell_elevation not in final_depths:
            final_depths[cell_elevation] = []
        flood_depth = final_H - cell_elevation
        if flood_depth >= 0:  # 침수 심도가 0 이상인 경우만 추가
            final_depths[cell_elevation].append(flood_depth)
    
    # 최종 침수 깊이 값 출력
    print(f"최종 침수 깊이 (H) 값: {final_H}")
    
    # 결과 시각화 및 저장
    visualize_and_save_flooded_area(grid_array, flooded_cells, sample_idx, final_H, final_depths)

if __name__ == "__main__":
    main(sample_idx=1)
