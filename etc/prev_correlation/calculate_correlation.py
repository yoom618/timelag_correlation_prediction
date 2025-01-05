import pandas as pd
from scipy.signal import correlate
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

# 1. CSV 파일 로드
def load_csv(file_path):
    data = pd.read_csv(file_path)
    if 'series1' not in data.columns or 'series2' not in data.columns:
        raise ValueError("CSV 파일은 'series1', 'series2' 열을 포함해야 합니다.")
    return data['series1'], data['series2']



def normalized_cross_correlation(x, y):
    """
    scipy.signal.correlate를 사용하여 정규화된 크로스 코릴레이션 계산
    """
    # 정규화된 입력 신호
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)

    # 크로스 코릴레이션 계산
    corr = correlate(x, y, mode='full')

    # 정규화된 크로스 코릴레이션
    corr /= len(x)
    return corr, np.arange(-len(x) + 1, len(x))

def dtw(x, y):
    n, m = len(x), len(y)
    cost = np.zeros((n, m))

    # 거리 행렬 계산
    for i in range(n):
        for j in range(m):
            cost[i, j] = abs(x[i] - y[j])

    # 누적 비용 행렬 초기화
    acc_cost = np.zeros((n, m))
    acc_cost[0, 0] = cost[0, 0]

    for i in range(1, n):
        acc_cost[i, 0] = cost[i, 0] + acc_cost[i-1, 0]

    for j in range(1, m):
        acc_cost[0, j] = cost[0, j] + acc_cost[0, j-1]

    for i in range(1, n):
        for j in range(1, m):
            acc_cost[i, j] = cost[i, j] + min(acc_cost[i-1, j], acc_cost[i, j-1], acc_cost[i-1, j-1])

    # 최적 경로 계산
    path = []
    i, j = n-1, m-1
    path.append((i, j))
    while i > 0 and j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            if acc_cost[i-1, j] == min(acc_cost[i-1, j], acc_cost[i, j-1], acc_cost[i-1, j-1]):
                i -= 1
            elif acc_cost[i, j-1] == min(acc_cost[i-1, j], acc_cost[i, j-1], acc_cost[i-1, j-1]):
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))
    path.reverse()

    return acc_cost[-1, -1], path

# 2. 상관계수 및 교차상관 함수
def calculate_correlation(series1, series2):
    correlation = np.corrcoef(series1, series2)[0, 1]
    return correlation


def calculate_cross_correlation(series1, series2):
    cross_corr = np.correlate(series1 - np.mean(series1), series2 - np.mean(series2), mode='full')
    return cross_corr

def standardize(series):
    return (series - np.mean(series)) / np.std(series)

# 3. 결과 시각화
def cross_correlation(series1, series2):

    # 교차상관 결과
    correlation, lags = normalized_cross_correlation(series1["close"], series2['close'])

    # 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.stem(lags, correlation)
    plt.title("Normalized Cross-Correlation with SciPy")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.grid()
    plt.savefig('/Users/hyeongigim/Desktop/projects/correlation/image/cross_correlation_image.png', dpi=300, bbox_inches='tight')

    plt.show()

def visualization(series1, series2):

    # 원본 시계열
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.plot(series1['date'], series1['close'], label='exchange')
    plt.plot(series2['date'], series2['close'], label='gole')
    plt.title('Time Series Data')
    plt.xlabel('Date')
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # 월별 주요 틱 설정
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # YYYY-MM 형식으로 레이블 표시
    plt.xticks(rotation=45)
    plt.savefig('/Users/hyeongigim/Desktop/projects/correlation/image/visualization_image.png', dpi=300,bbox_inches='tight')
    plt.show()

def granger_causality_test(x, y):
    print("귀무가설1 : Series1의 데이터는 Series2의 데이터를 그레인저 인과하지 않는다")

    data = pd.DataFrame({'X': x, 'Y': y})
    data.to_csv("./result.csv")
    result = grangercausalitytests(data[['Y', 'X']], maxlag=12)
    min = 1
    min_leg = 0
    for lag, values in result.items():
        print(f"Lag {lag}:")
        print(f"  F-statistic: {values[0]['ssr_ftest'][0]}")
        print(f"  p-value: {values[0]['ssr_ftest'][1]}")
        if values[0]['ssr_ftest'][1] < 0.05:
            min = values[0]['ssr_ftest'][1]
            min_leg = lag
    if min < 0.05:
        print(f"p-value {min} < 0.05 in leg {min_leg} 이므로 귀무가설1을 기각한다.")
    else:
        print(f"p-value {min} > 0.05 in leg {min_leg}.")

    print("귀무가설2 : Series2의 데이터는 Series1의 데이터를 그레인저 인과하지 않는다")
    result = grangercausalitytests(data[['X', 'Y']], maxlag=12)
    min = 1
    min_leg = 0
    for lag, values in result.items():
        print(f"Lag {lag}:")
        print(f"  F-statistic: {values[0]['ssr_ftest'][0]}")
        print(f"  p-value: {values[0]['ssr_ftest'][1]}")
        if values[0]['ssr_ftest'][1] < 0.05:
            min = values[0]['ssr_ftest'][1]
            min_leg = lag
    if min < 0.05:
        print(f"p-value {min:.4f} < 0.05 in leg {min_leg} 이므로 귀무가설2를 기각한다.")
    else:
        print(f"p-value {min:.4f} > 0.05 in leg {min_leg}.")


def analyze_correlation(file1_path, file2_path,start_date, end_date):
    series1 = pd.read_csv(file1_path)
    series2 = pd.read_csv(file2_path,encoding='cp949')

    #날짜 처리
    series1['date'] = pd.to_datetime(series1['date'])
    series2['date'] = pd.to_datetime(series2['date'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    series1 = series1[(series1['date'] >= start_date) & (series1['date'] <= end_date)]
    series2 = series2[(series2['date'] >= start_date) & (series2['date'] <= end_date)]

    #표준화
    series1['close'] = standardize(series1['close'])
    series2['close'] = standardize(series2['close'])
    x, y = np.array(series1['close']), np.array(series2['close'])

    #시각화
    visualization(series1, series2)
    # 피어슨 상관계수
    correlation = calculate_correlation(x, y)
    print(f"Pearson Correlation Coefficient: {correlation:.2f}")

    # 스피어만 상관계수
    correlation, p_value = spearmanr(x, y)
    print(f"Spearman Correlation Coefficient: {correlation:.2f}")

    # 크로스 상관계수
    cross_correlation(series1, series2)

    #DTW
    distance, path = dtw(x, y)
    print(f"DTW Distance: {distance}")
    plt.imshow(cdist(x.reshape(-1, 1), y.reshape(-1, 1)), cmap='gray', origin='lower')
    for (i, j) in path:
        plt.plot(j, i, 'ro')
    plt.title('DTW Optimal Path')
    plt.xlabel('Sequence Y')
    plt.ylabel('Sequence X')
    plt.savefig('/Users/hyeongigim/Desktop/projects/correlation/image/DTW_image.png', dpi=300, bbox_inches='tight')
    plt.show()

    #그레인저 인과검정
    granger_causality_test(x, y)


if __name__ == '__main__':
    # 파일 경로 입력
    file1_path = '/Users/hyeongigim/Desktop/projects/correlation/exchange.csv'  # CSV 파일 경로
    file2_path = '/Users/hyeongigim/Desktop/projects/correlation/gold_data.csv'  # CSV 파일 경로
    start_date = '2024-01-01' # 분석 시작 날짜
    end_date = '2024-11-22' # 분석 끝 날짜
    analyze_correlation(file1_path, file2_path,start_date,end_date)



