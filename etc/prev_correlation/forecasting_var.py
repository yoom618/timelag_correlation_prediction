import pandas as pd
from statsmodels.tsa.api import VAR
import numpy as np
import matplotlib.pyplot as plt



def standardize(series):
    return (series / series.iloc[0]) - 1


def make_df(file1_path, file2_path, start_date, end_date):
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
    series1['close1'] = standardize(series1['close'])
    series2['close2'] = standardize(series2['close'])
    
    df = pd.concat([series1['close1'],series2['close2']],axis = 1)
    return df

def var_analysis(df):
    pass



if __name__ == '__main__':
    # 파일 경로 입력
    file1_path = '/Users/hyeongigim/Desktop/projects/correlation/exchange.csv'  # CSV 파일 경로
    file2_path = '/Users/hyeongigim/Desktop/projects/correlation/gold_data.csv'  # CSV 파일 경로
    start_date = '2023-11-22' # 분석 시작 날짜
    end_date = '2024-11-22' # 분석 끝 날짜
    df = make_df(file1_path, file2_path, start_date, end_date)
    df.to_csv("./result.csv")

    
    # VAR 모델 훈련
    model = VAR(df)
    results = model.fit(maxlags=15, ic='aic')
    
    # 예측
    forecast = results.forecast(df.values[-results.k_ar:], steps=5)
    print(forecast)
    # IRF 계산
    irf = results.irf(10)  # 10시간 간격까지의 반응을 계산

    # IRF 결과 시각화
    fig = irf.plot(orth=True)  # Orthogonalized IRF
    # 그래프 타이틀 설정 (선택 사항)
    fig.suptitle("Impulse Response Function", fontsize=16)

    # 그래프 저장 (선택 사항)
    plt.savefig("irf_plot.png")

    # 그래프 표시
    plt.show()
    # IRF 값을 데이터프레임으로 변환
    irf_values = irf.irfs
    

