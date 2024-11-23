from urllib.request import urlopen
from datetime import datetime, timedelta
from pytz import timezone
import numpy as np

# 한국 시간대 설정
kst = timezone("Asia/Seoul")

# API 정보
API_URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min?"
API_KEY = "ojMlPtAdS2WzJT7QHdtlwg"  # 인증 키
STATION_ID = "401"  # 관측소 ID
DISP = "0"  # 데이터 반환 형식 (CSV 형태)
HELP = "1"  # 도움말 비활성화

def fetch_aws_data(start_time, end_time):
    """
    AWS API를 호출하여 주어진 시간 범위의 데이터를 가져옵니다.
    """
    tm1 = f"tm1={start_time.strftime('%Y%m%d%H%M')}"
    tm2 = f"tm2={end_time.strftime('%Y%m%d%H%M')}"
    stn = f"stn={STATION_ID}"
    disp = f"disp={DISP}"
    help_flag = f"help={HELP}"
    auth = f"authKey={API_KEY}"
    
    full_url = f"{API_URL}{tm1}&{tm2}&{stn}&{disp}&{help_flag}&{auth}"

    print(f"API 호출 URL: {full_url}")

    try:
        # API 호출
        with urlopen(full_url) as response:
            # EUC-KR 인코딩으로 응답을 디코딩
            data = response.read().decode('euc-kr')
            print(f"응답 데이터:\n{data}")

            # 데이터가 비어 있는지 확인
            if "준비중입니다" in data or "잘못 입력" in data:
                raise ValueError("API URL이 올바르지 않거나 데이터가 존재하지 않습니다.")

            # 데이터 처리
            lines = data.split("\n")
            rainfall_data = {}
            for line in lines:
                if line.strip() and not line.startswith("#"):  # 데이터 줄 필터링
                    columns = line.split()  # 공백을 기준으로 분리
                    if len(columns) >= 13:  # 최소 13열 필요
                        timestamp = datetime.strptime(columns[0], "%Y%m%d%H%M").replace(tzinfo=kst)
                        rn_day = float(columns[12])  # 13번째 열: RN-DAY
                        rainfall_data[timestamp] = rn_day

            if not rainfall_data:
                raise ValueError("RN-DAY 데이터를 찾을 수 없습니다.")
            
            return rainfall_data

    except Exception as e:
        print(f"오류 발생: {e}")
        return {}

def calculate_10min_rainfall(rainfall_data, target_times):
    """
    주어진 시간에 대해 10분 누적 강수량을 계산합니다.
    """
    ten_min_rainfalls = []
    for t in target_times:
        t_minus_10 = t - timedelta(minutes=10)
        if t in rainfall_data and t_minus_10 in rainfall_data:
            ten_min_rainfall = max(0, rainfall_data[t] - rainfall_data[t_minus_10])
            ten_min_rainfalls.append((t.strftime("%Y-%m-%d %H:%M"), ten_min_rainfall))
        else:
            ten_min_rainfalls.append((t.strftime("%Y-%m-%d %H:%M"), None))  # 데이터가 없을 경우 None 처리
    return ten_min_rainfalls

def generate_numpy_arrays(ten_min_rainfalls, grid_shape=(64, 64, 1)):
    """
    10분 누적 강수량 데이터를 (64, 64, 1) 형태의 numpy 배열로 변환하고,
    이를 합쳐 (4, 64, 64, 1) 형태의 numpy 배열로 생성합니다.
    """
    arrays = []
    for _, value in ten_min_rainfalls:
        if value is not None:
            # (64, 64, 1) 형태로 생성
            grid_array = np.full(grid_shape, value, dtype=np.float32)
            arrays.append(grid_array)
        else:
            # 데이터가 없으면 0으로 채움
            grid_array = np.full(grid_shape, 0, dtype=np.float32)
            arrays.append(grid_array)

    # (4, 64, 64, 1) 형태로 변환
    final_array = np.stack(arrays, axis=0)
    return final_array

def main():
    """
    2022년 8월 19일 16:40부터 17:20까지의 데이터를 가져오고,
    주어진 시점의 10분 누적 강수량을 계산합니다.
    """
    # 시작 시간과 끝 시간 설정
    start_time = datetime(2022, 8, 19, 16, 40, tzinfo=kst)
    end_time = datetime(2022, 8, 19, 17, 20, tzinfo=kst)

    # 타겟 시간 (10분 간격)
    target_times = [
        datetime(2022, 8, 19, 16, 50, tzinfo=kst),
        datetime(2022, 8, 19, 17, 0, tzinfo=kst),
        datetime(2022, 8, 19, 17, 10, tzinfo=kst),
        datetime(2022, 8, 19, 17, 20, tzinfo=kst),
    ]

    # AWS 데이터 가져오기
    rainfall_data = fetch_aws_data(start_time, end_time)
    print(f"RN-DAY 데이터: {rainfall_data}")

    # 10분 누적 강수량 계산
    ten_min_rainfalls = calculate_10min_rainfall(rainfall_data, target_times)
    for time, value in ten_min_rainfalls:
        if value is not None:
            print(f"{time} - 10분 누적 강수량: {value:.1f} mm")
        else:
            print(f"{time} - 데이터 없음")

    # numpy 배열 생성
    final_array = generate_numpy_arrays(ten_min_rainfalls)
    print(f"최종 numpy 배열 형태: {final_array.shape}")
    print(f"최종 numpy 배열 내용:\n{final_array}")

    # numpy 파일 저장
    np.save("rainfall_data.npy", final_array)
    print("numpy 배열이 'rainfall_data.npy'로 저장되었습니다.")

if __name__ == "__main__":
    main()
