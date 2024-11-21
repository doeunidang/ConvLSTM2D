import openpyxl
import os
from openpyxl.utils.dataframe import dataframe_to_rows
from pyswmm import Simulation, Nodes
import pandas as pd
from datetime import timedelta

# INP 파일이 있는 폴더와 Excel 파일을 저장할 폴더 경로 설정
inp_folder = "F:\\ConvLSTM2D_git\\ConvLSTM2D\\SWMM_inp\\INP"
output_folder = "F:\\ConvLSTM2D_git\\ConvLSTM2D\\DATA_goal"

# Junction ID 리스트 (J1부터 J30까지)
junction_ids = [f'J{i}' for i in range(1, 34)]

# INP 폴더에서 파일 목록 가져오기
inp_files = [f for f in os.listdir(inp_folder) if f.endswith('.inp')]

# INP 파일마다 시뮬레이션 실행 및 Excel 파일 생성
for inp_file in inp_files:
    file_number = inp_file.split('_')[-1].split('.')[0]  # test_n에서 n 추출
    inp_file_path = os.path.join(inp_folder, inp_file)
    
    # 엑셀 파일 경로 설정
    output_excel = os.path.join(output_folder, f'Junction_Flooding_{file_number}.xlsx')

    # 시뮬레이션 실행
    with Simulation(inp_file_path) as sim:
        nodes = Nodes(sim)
        results = []

        # 시뮬레이션을 10분 단위로만 진행하도록 step 간격 설정
        sim.step_advance(600)  # 600초 = 10분

        # 시뮬레이션 시작 시간 확인
        start_time = sim.start_time
        next_record_time = start_time + timedelta(minutes=10 - (start_time.minute % 10), seconds=-start_time.second, microseconds=-start_time.microsecond)

        for step in sim:
            current_time = sim.current_time

            # 10분 간격으로 기록 (기록 시점에 도달했을 때)
            if current_time >= next_record_time:
                row = [current_time]
                for junction_id in junction_ids:
                    node = nodes[junction_id]
                    # flooding 값을 소숫점 둘째 자리까지만 표시하고 float 형식으로 저장
                    row.append(round(node.flooding, 3))
                results.append(row)
                # 다음 기록 시간을 10분 후로 설정
                next_record_time += timedelta(minutes=10)

        # 시뮬레이션 마지막 시간도 강제로 기록
        current_time = sim.current_time
        row = [current_time]
        for junction_id in junction_ids:
            node = nodes[junction_id]
            row.append(round(node.flooding, 2))
        results.append(row)

    # 결과를 데이터프레임으로 변환
    df = pd.DataFrame(results, columns=['Time'] + junction_ids)

    # 엑셀 파일로 저장
    with pd.ExcelWriter(output_excel, engine='openpyxl', datetime_format='YYYY-MM-DD HH:MM:SS') as writer:
        df.to_excel(writer, index=False)

        # 셀 크기 자동 조정
        workbook = writer.book
        worksheet = workbook.active

        # 각 열의 넓이를 내용에 맞게 자동 조정
        for column in worksheet.columns:
            max_length = 0
            column = list(column)
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = (max_length + 2)
            worksheet.column_dimensions[column[0].column_letter].width = adjusted_width

        # 날짜 및 시간 형식을 제대로 설정
        for cell in worksheet['A']:  # 'Time' 열
            cell.number_format = 'YYYY-MM-DD HH:MM:SS'
        
        # 숫자 형식 적용 (나머지 열)
        for col in worksheet.iter_cols(min_col=2, min_row=2, max_col=len(junction_ids)+1):
            for cell in col:
                cell.number_format = '0.000'  # 소수점 두 자리 형식

    print(f"{output_excel} 파일이 생성되었습니다.")


