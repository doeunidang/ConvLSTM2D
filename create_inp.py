import geopandas as gpd
import os

# 파일 경로 설정 (폴더 경로들)
subcatchment_shp = '.\\SWMM_inp\\SHP\\subcatchment.shp'
junction_shp = '.\\SWMM_inp\\SHP\\junction.shp'
conduit_shp = '.\\SWMM_inp\\SHP\\conduit.shp'
outfall_shp = '.\\SWMM_inp\\SHP\\outfall.shp'
pump_shp = '.\\SWMM_inp\\SHP\\pump.shp'
rainfall_folder = 'F:\\ConvLSTM2D_git\\ConvLSTM2D\\DATA_input\\RAINFALL'
inp_output_folder = 'F:\\ConvLSTM2D_git\\ConvLSTM2D\\SWMM_inp\\INP'

# EPSG:5186 좌표계를 설정합니다.
target_crs = 'EPSG:5186'

# 1. Subcatchments 읽기 (유역 정의)
subcatchments = gpd.read_file(subcatchment_shp)
if subcatchments.crs != target_crs:
    subcatchments = subcatchments.to_crs(target_crs)

# 2. Junctions 읽기 (교차점 정의)
junctions = gpd.read_file(junction_shp)
if junctions.crs != target_crs:
    junctions = junctions.to_crs(target_crs)

# 3. Conduits 읽기 (관로 정의)
conduits = gpd.read_file(conduit_shp)
if conduits.crs != target_crs:
    conduits = conduits.to_crs(target_crs)

# 4. Outfall 포인트 레이어 읽기
outfalls = gpd.read_file(outfall_shp)
if outfalls.crs != target_crs:
    outfalls = outfalls.to_crs(target_crs)

# 5. Pump 읽기 (펌프 정의)
pumps = gpd.read_file(pump_shp)
if pumps.crs != target_crs:
    pumps = pumps.to_crs(target_crs)

# Rainfall 파일 리스트 가져오기
rainfall_files = [f for f in os.listdir(rainfall_folder) if f.endswith('.dat')]
rainfall_files.sort()  # 파일 이름 순서대로 정렬

# INP 파일 생성
for rainfall_file in rainfall_files:
    file_number = rainfall_file.split('_')[-1].split('.')[0]  # rainfall_event_x에서 x 추출

    inp_output_path = os.path.join(inp_output_folder, f'test_{file_number}.inp')
    rainfall_file_path = os.path.join(rainfall_folder, rainfall_file)
    
    with open(rainfall_file_path, 'r') as rf:
        lines = rf.readlines()
        first_line = lines[0].strip().split()
        last_line = lines[-1].strip().split()

        start_year = first_line[1]
        start_month = first_line[2].zfill(2)
        start_day = first_line[3].zfill(2)
        start_hour = first_line[4].zfill(2)
        start_minute = first_line[5].zfill(2)

        end_year = last_line[1]
        end_month = last_line[2].zfill(2)
        end_day = last_line[3].zfill(2)
        end_hour = last_line[4].zfill(2)
        end_minute = last_line[5].zfill(2)

    with open(inp_output_path, 'w') as f:
        # [OPTIONS] 섹션 작성
        f.write('[OPTIONS]\n')
        f.write('FLOW_UNITS           CMS\n')
        f.write(f'START_DATE           {start_month}/{start_day}/{start_year}\n')
        f.write(f'START_TIME           {start_hour}:{start_minute}:00\n')
        f.write(f'REPORT_START_DATE    {start_month}/{start_day}/{start_year}\n')
        f.write(f'REPORT_START_TIME    {start_hour}:{start_minute}:00\n')
        f.write(f'END_DATE             {end_month}/{end_day}/{end_year}\n')
        f.write(f'END_TIME             {end_hour}:{end_minute}:00\n')
        f.write('SWEEP_START          01/01\n')
        f.write('SWEEP_END            12/31\n')
        f.write('DRY_DAYS             0\n')
        f.write('REPORT_STEP          00:10:00\n')
        f.write('WET_STEP             00:05:00\n')
        f.write('DRY_STEP             01:00:00\n')
        f.write('ROUTING_STEP         00:05:00\n')
        f.write('ALLOW_PONDING        NO\n')
        f.write('INERTIAL_DAMPING     NONE\n')
        f.write('FORCE_MAIN_EQUATION  H-W\n')
        f.write('LINK_OFFSETS         DEPTH\n')
        f.write('MIN_SLOPE            0\n')

        # [SUBCATCHMENTS] 섹션 작성
        f.write('\n[SUBCATCHMENTS]\n')
        f.write(';;Name           Raingage         Outlet          Area       %Imperv   Width     Slope     CurbLen   SnowPack\n')
        for idx, row in subcatchments.iterrows():
            subcatchment_id = row['gid']
            outlet = row['Outlet']
            imperv = row['%Imperv']
            slope = row['%Slope']
            area = row['Area_ha']
            f.write(f"{subcatchment_id}          R{file_number}          {outlet}         {area:.3f}     {imperv:.2f}       500       {slope:.2f}      0        \n")

       # Subareas 작성
        f.write('\n[SUBAREAS]\n')
        f.write(';;Subcatchment     N-Imperv   N-Perv     S-Imperv    S-Perv     PctZero    RouteTo     PctRouted\n')
        
        for idx, row in subcatchments.iterrows():
            subcatchment_id = row['gid']  # Subcatchment ID
            n_imperv = row['N_Imperv']  # Manning's n for Impervious Areas
            n_perv = row['N_perv']  # Manning's n for Pervious Areas
            
            # Subarea 작성
            f.write(f"{subcatchment_id}          {n_imperv:.4f}       {n_perv:.4f}       0.05        0.05       25         OUTLET      100\n")
        
        # Polygons 섹션 작성 (각 유역의 경계 좌표를 SWMM에 맞게 추가 가능)
        f.write('\n[POLYGONS]\n')
        f.write(';;Subcatchment     X-Coord         Y-Coord\n')
        
        for idx, row in subcatchments.iterrows():
            subcatchment_id = row['gid']  # Subcatchment ID (QGIS의 'gid'로 변경)
            geometry = row['geometry']
            
            # 폴리곤 처리
            if geometry.geom_type == 'Polygon':
                for x, y in geometry.exterior.coords:
                    f.write(f"{subcatchment_id}         {x:.3f}           {y:.3f}\n")
            elif geometry.geom_type == 'MultiPolygon':
                for poly in geometry.geoms:
                    for x, y in poly.exterior.coords:
                        f.write(f"{subcatchment_id}         {x:.3f}           {y:.3f}\n")

        # Junctions Shapefile에서 읽어온 Junction 작성
        f.write('\n[JUNCTIONS]\n')
        f.write(';;Name           Elevation    MaxDepth    InitDepth    SurDepth    Aponded\n')
        
        junction_ids = set()  # 모든 junction ID를 저장할 집합
        for idx, row in junctions.iterrows():
            junction_id = row['id']  # Junction ID
            junction_ids.add(junction_id)  # junction ID 추가
            max_depth = row['max_depth']  # Max Depth
            ini_depth = row['ini_depth']  # Initial Depth
            
            # Junction 작성 (이제 Subcatchment 중심이 아닌, Junction Shapefile에 있는 데이터를 사용)
            f.write(f"{junction_id}          0           {max_depth:.3f}           {ini_depth:.3f}           0           0\n")
        
        # Junction 좌표 추가
        f.write('\n[COORDINATES]\n')
        f.write(';;Node           X-Coord           Y-Coord\n')
        for idx, row in junctions.iterrows():
            junction_id = row['id']
            x, y = row['geometry'].x, row['geometry'].y  # Junction Shapefile에서 X, Y 좌표 추출
            f.write(f"{junction_id}           {x:.3f}           {y:.3f}\n")

        # [PUMPS] 섹션 작성 (with Pump Curve as '*', and Initial Status ON)
        f.write('\n[PUMPS]\n')
        f.write(';;Name           InletNode      OutletNode     PumpCurve   InitialStatus   StartupDepth   ShutoffDepth\n')
        for idx, row in pumps.iterrows():
            pump_id = row['name']  # 펌프 ID
            geometry = row['geometry']
            if geometry.geom_type == 'LineString':
                inlet_coords = geometry.coords[0]  # 라인의 첫 좌표
                outlet_coords = geometry.coords[-1]  # 라인의 마지막 좌표

                # 펌프의 Inlet과 Outlet을 정의합니다.
                inlet = 'J31'  # 예시로 지정된 inlet junction
                outlet = 'J19'  # 예시로 지정된 outlet junction
                startup_depth = 10.4  # Startup Depth
                shutoff_depth = 6.6  # Shutoff Depth

                # 펌프 커브는 '*'로 설정, Initial Status는 'ON'
                f.write(f"{pump_id}          {inlet}         {outlet}         *         ON          {startup_depth}          {shutoff_depth}\n")

        # Outfalls 섹션 작성
        f.write('\n[OUTFALLS]\n')
        f.write(';;Name           Elevation   Type       Outfall_Units\n')
        
        outfall_ids = set()  # Outfall ID를 저장할 집합
        for idx, row in outfalls.iterrows():
            outfall_id = row['id']  # Outfall ID
            outfall_ids.add(outfall_id)
            elevation = 0.0
            f.write(f"{outfall_id}          {elevation:.3f}        FREE       NO\n")

        # Outfalls 좌표 추가
        f.write('\n[COORDINATES]\n')
        f.write(';;Node           X-Coord           Y-Coord\n')
        for idx, row in outfalls.iterrows():
            outfall_id = row['id']
            x, y = row['geometry'].x, row['geometry'].y  # Outfall Shapefile에서 X, Y 좌표 추출
            f.write(f"{outfall_id}           {x:.3f}           {y:.3f}\n")

        # Conduits Shapefile에서 읽어온 관로 작성
        f.write('\n[CONDUITS]\n')
        f.write(';;Name           Inlet Node        Outlet Node     Length      Roughness   InOffset  OutOffset  InitFlow  MaxFlow\n')
            
        conduits_defined = set()
        for idx, row in conduits.iterrows():
            conduit_id = row['id']  # Conduit ID
            inlet = row['inlet']  # Inlet Node ID
            outlet = row['outlet']  # Outlet Node ID
            length = row['length']  # Length
            roughness = row['roughness']  # Manning's roughness coefficient

            # Conduit 작성 전에 Junction과 Outfall이 정의되어 있는지 확인
            if inlet not in junction_ids and inlet not in outfall_ids:
                print(f"오류: Inlet Node '{inlet}'이(가) 정의되지 않았습니다.")
            if outlet not in junction_ids and outlet not in outfall_ids:
                print(f"오류: Outlet Node '{outlet}'이(가) 정의되지 않았습니다.")
            
            conduits_defined.add(conduit_id)
            f.write(f"{conduit_id}          {inlet}          {outlet}          {length:.3f}       {roughness:.3f}       0          0         0         0\n")

        # Vertices 섹션 작성 (각 conduit의 경로)
        f.write('\n[VERTICES]\n')
        f.write(';;Link           X-Coord           Y-Coord\n')
        
        for idx, row in conduits.iterrows():
            conduit_id = row['id']
            geometry = row['geometry']
            
            for x, y in geometry.coords:
                f.write(f"{conduit_id}           {x:.3f}           {y:.3f}\n")

        # [XSECTIONS] 섹션 추가
        f.write('\n[XSECTIONS]\n')
        f.write(';;Link            Shape        Geom1     Geom2     Geom3     Geom4     Barrels\n')
        for conduit_id in conduits_defined:
            f.write(f"{conduit_id}         CIRCULAR    1.65       0.0       0.0       0.0       1\n")

        # [RAINGAGES] 섹션 작성
        rain_gage_id = f'R{file_number}'
        station_id = file_number
        f.write('\n[RAINGAGES]\n')
        f.write(';;Name           Interval     SCF        Source    File Name    Station ID    Rain Units\n')
        f.write(f"{rain_gage_id}     VOLUME    00:10     1.0       FILE      {rainfall_file_path}          {station_id}        IN\n")

print(f"INP 파일이 {inp_output_folder}에 성공적으로 생성되었습니다.")
