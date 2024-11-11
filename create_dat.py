import pandas as pd
from datetime import timedelta

# 엑셀 파일 불러오기
file_path = 'F:\\ConvLSTM2D_git\\ConvLSTM2D\\SWMM_inp\\강우사상_300개.xlsx'
df = pd.read_excel(file_path)

# 엑셀 파일의 열 이름 출력 (강우사상이 있는 열을 확인)
print("엑셀 파일의 열 이름:", df.columns)

# 엑셀 파일의 강우사상이 시작되는 열 인덱스 (예: B열부터 각 강우사상이 열로 정의)
rainfall_start_idx = 1  # B열이 1번째 열이라고 가정

# 각 강우사상별 시작 날짜와 시간이 정의된 행 번호 설정 (DATE가 있는 행 번호)
date_row = 1  # 2번째 행에 DATE가 있으므로 이 값으로 설정

# 간격 설정 (예: 10분 간격)
interval = timedelta(minutes=10)

# Gauge 번호를 1부터 시작하여 각 강우사상에 대해 1씩 증가하도록 설정
gauge_number = 1

# 각 강우사상 데이터를 DAT 파일로 저장
dat_files = []  # 생성된 dat 파일들을 추적하기 위한 리스트
for i in range(rainfall_start_idx, len(df.columns)):  # 각 열을 순회 (강우사상별)
    # 각 열의 시작 날짜/시간을 가져오기 (DATE 셀이 있는 위치를 기준으로)
    start_time = pd.to_datetime(df.iloc[date_row, i], errors='coerce')  # 날짜/시간이 있는 셀 가져오기
    
    # 날짜/시간 값이 없거나 NaT인 경우 프로그램 종료
    if pd.isna(start_time):
        print(f"열 {i}에서 날짜/시간 값이 없거나 잘못된 값이 있습니다. 프로그램을 중단합니다.")
        break  # 날짜가 없으면 프로그램 종료

    # 각 강우사상의 강우량 데이터 추출 (빈 값 및 마지막 합계(Sum) 행 제외)
    rainfall_data = df.iloc[2:-1, i].dropna()  # 3번째 행부터 시작, 마지막 행은 제외

    # 데이터를 저장할 빈 리스트 생성
    data = []

    # 강우사상 데이터를 10분 간격으로 시간, 값과 함께 저장
    for idx, rainfall in enumerate(rainfall_data):
        time = start_time + idx * interval
        # Gauge, Year, Month, Day, Hour, Min, Value 형식으로 저장
        data.append([
            gauge_number,                     # Gauge 번호 (각 강우사상마다 1씩 증가)
            time.year,                        # Year
            time.month,                       # Month
            time.day,                         # Day
            time.hour,                        # Hour
            time.minute,                      # Minute
            round(rainfall, 3)                # Value (소수점 3자리까지)
        ])

    # DataFrame으로 변환
    df_new = pd.DataFrame(data, columns=['Gauge', 'Year', 'Month', 'Day', 'Hour', 'Min', 'Value'])
    
    # DAT 파일로 저장 (각 강우사상에 대해 파일명을 다르게 설정)
    output_dat_path = f'F:\\ConvLSTM2D_git\\ConvLSTM2D\\DATA_input\\RAINFALL\\rainfall_event_{i}.dat'
    df_new.to_csv(output_dat_path, index=False, header=False, sep='\t')  # 탭으로 구분된 형식으로 저장

    dat_files.append(output_dat_path)  # 생성된 DAT 파일 경로를 리스트에 추가
    print(f"DAT 파일이 {output_dat_path}에 성공적으로 저장되었습니다.")
    
    # 다음 강우사상에 대해 Gauge 번호를 1씩 증가
    gauge_number += 1

# 생성된 DAT 파일 중 하나의 내용을 확인
if dat_files:
    file_to_check = dat_files[250]  # 첫 번째로 생성된 파일을 확인
    print(f"\n{file_to_check} 파일 내용:")
    with open(file_to_check, 'r') as f:
        print(f.read())  # 파일 내용을 출력
