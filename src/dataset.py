# dataset.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import get_data_path, encode_base64
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# FutureWarning 제거
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


SYMBOLS = sorted(['BrentOil', 'Copper', 'CrudeOil', 'Gasoline', 
                  'Gold', 'NaturalGas', 'Platinum', 'Silver',
                  'AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY', 'USD'])

class PriceDataset(Dataset):
    def __init__(self, 
                 start_date, 
                 end_date, 
                 phase='train',
                 in_columns=['USD_Price'], 
                 out_columns=['USD_Price'],
                 input_days=3, 
                 time_lag=1,
                 data_dir='data', 
                 **kwargs):
        self.x, self.y, self.x_index, self.y_index, self.scaler_x, self.scaler_y = \
            make_features(start_date, end_date, phase,
                         in_columns, out_columns, input_days, time_lag,
                         data_dir, **kwargs)
    
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_features(start_date, end_date, phase, in_columns, out_columns, input_days, time_lag,
                  data_dir='data', **kwargs):
    
    ##### 입력값 검증
    assert input_days > 0, 'input_days should be greater than 0'
    assert time_lag > 0, 'time_lag should be greater than 0'
    assert len(in_columns) > 0, 'in_columns should have at least one column'
    assert len(out_columns) > 0, 'out_columns should have at least one column'

    ##### 데이터 불러오기
    # preprocessed 폴더에 저장된 메타데이터를 불러옴
    symbols = set([col.split('_')[0] for col in in_columns + out_columns])
    table = load_dataset(start_date, end_date, data_dir, symbols)

    # 데이터 기간 설정
    table = table.loc[start_date:end_date]
    print(f'\t-> Selected Data Period: {start_date} - {end_date}\n')


    ##### TODO: 데이터 클렌징 및 전처리
    # 주의 : out_columns에는 값이 있고, 나머지에는 값이 없는 경우가 있음. 이러한 경우는 삭제되지 않도록 주의할 것
    #       이를 제거할지 여부는 사용자의 판단에 따라 결정. 여기서는 앞선 데이터로 대체하도록 함
    table = table.dropna(subset=out_columns, how='any')  # out_columns에 하나라도 결측치가 있는 경우 해당 row 삭제
    table = table.interpolate(method='linear')  # 결측치를 선형적으로 보간
    table = table.ffill().bfill()  # 나머지 결측치를 앞선 값으로 대체

    use_columns = list(set(in_columns + out_columns))  # 중복 제거
    df = table[use_columns]
    del table  # 메모리 절약을 위해 사용하지 않는 변수 삭제

    ##### TODO: 추가적인 feature engineering이 필요하다면 아래에 작성
    # 가령, 주식 데이터의 경우 이동평균선, MACD, RSI 등의 feature를 생성할 수 있음
    # 주의 : 미래 데이터를 활용하는 일이 없도록 유의할 것 (가령, 10월 31일 데이터(row)에 10월 31일 뒤의 데이터가 활용되면 안 됨)
    # 주의 : 추가로 활용할 feature들은 in_columns에도 추가할 것
    in_columns += []


    # out_column이 아닌 in_column에 대해서는 time_lag를 적용
    # 가령, Gold+USD -> USD의 경우, Gold 데이터만 time_lag를 적용하고, USD는 적용하지 않음
    lag_columns = list(set(in_columns) - set(out_columns))
    df_lagged = df.copy()
    df_lagged[lag_columns] = df_lagged[lag_columns].shift(time_lag)  # time_lag 이전의 데이터로 대체
    df_lagged = df_lagged.fillna(method='bfill')  # 0 ~ time_lag-1번째까지는 데이터가 없으므로 이를 채워줌

    # x에 대해서는 input_days만큼의 데이터를 묶어서 feature로 사용
    # y에 대해서는 1일 뒤의 데이터를 예측 대상으로 사용
    date_indices = [idx.strftime("%Y-%m-%d") for idx in df.index]
    date_indices = [date_indices[0],]*time_lag + date_indices
    date_indices_lagged = [idx.strftime("%Y-%m-%d") for idx in df_lagged.index]
    x = np.asarray([df_lagged.loc[date_indices_lagged[i:i + input_days], in_columns] 
                    for i in range(len(date_indices_lagged) - input_days)])
    y = np.asarray([df.loc[date_indices_lagged[i + input_days], out_columns] 
                    for i in range(len(date_indices_lagged) - input_days)])
    x_index = []
    for i in range(len(date_indices_lagged) - input_days):
        for j in range(input_days):
            for k in in_columns:
                if k in out_columns:
                    x_index.append(date_indices_lagged[i + j])
                else:
                    x_index.append(date_indices[i + j])
    x_index = np.asarray(x_index).reshape(-1, input_days, len(in_columns))

    y_index = date_indices_lagged[input_days:]


    # Data Split
    # 제공하는 타입 : all / train+valid / train / valid / test 
    # 여기서는 마지막 10일을 test 데이터로 사용하고, 나머지를 9:1 비율로 train / valid로 나눔
    # 이러한 파라미터는 **kwargs로 받아서 사용자가 변경할 수 있도록 함
    test_size = kwargs.get('test_size', 10)
    train_ratio = kwargs.get('train_ratio', 0.9)
    split = int((len(x) - test_size) * train_ratio)

    train_x, train_y = x[:split], y[:split]
    train_x_index, train_y_index = x_index[:split], y_index[:split]
    valid_x, valid_y = (x[split:-test_size], y[split:-test_size]) \
        if test_size > 0 else (x[split:], y[split:])
    valid_x_index, valid_y_index = (x_index[split:-test_size], y_index[split:-test_size]) \
        if test_size > 0 else (x_index[split:], y_index[split:])
    test_x, test_y = (x[-test_size:], y[-test_size:]) \
        if test_size > 0 else ([], [])
    test_x_index, test_y_index = (x_index[-test_size:], y_index[-test_size:]) \
        if test_size > 0 else ([], [])
    

    # 데이터 정규화
    # 이 때, 학습 및 검증 데이터에 대해서는 fit_transform을 사용하고, 테스트 데이터에 대해서는 transform만 사용
    scaler_x, scaler_y = None, None
    scaling = kwargs.get('scaling', False)
    if scaling:
        if scaling == 'minmax':
            scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
        elif scaling == 'standard':
            scaler_x, scaler_y = StandardScaler(), StandardScaler()
        else:
            raise ValueError(f'Unknown scaling method: {scaling}')
        scaler_x.fit(np.concatenate([train_x.reshape(-1, len(in_columns)), 
                                    valid_x.reshape(-1, len(in_columns))], axis=0))
        train_x = scaler_x.transform(train_x.reshape(-1, len(in_columns))).reshape(train_x.shape)
        valid_x = scaler_x.transform(valid_x.reshape(-1, len(in_columns))).reshape(valid_x.shape) \
            if len(valid_x) > 0 else None
        test_x = scaler_x.transform(test_x.reshape(-1, len(in_columns))).reshape(test_x.shape) \
            if len(test_x) > 0 else None
        
        train_y = scaler_y.fit_transform(train_y)
        valid_y = scaler_y.transform(valid_y) if len(valid_y) > 0 else None
        test_y = scaler_y.transform(test_y) if len(test_y) > 0 else None


    if phase == 'all':
        X = np.concatenate([train_x, valid_x, test_x], axis=0)
        Y = np.concatenate([train_y, valid_y, test_y], axis=0)
        x_index = np.concatenate([train_x_index, valid_x_index, test_x_index], axis=0)
        y_index = train_y_index + valid_y_index + test_y_index
    elif phase == 'train+valid':
        X = np.concatenate([train_x, valid_x], axis=0)
        Y = np.concatenate([train_y, valid_y], axis=0)
        x_index = np.concatenate([train_x_index, valid_x_index], axis=0)
        y_index = train_y_index + valid_y_index
    elif phase == 'train':
        X = train_x
        Y = train_y
        x_index = train_x_index
        y_index = train_y_index
    elif phase == 'valid':
        X = valid_x
        Y = valid_y
        x_index = valid_x_index
        y_index = valid_y_index
    elif phase == 'test':
        X = test_x
        Y = test_y
        x_index = test_x_index
        y_index = test_y_index


    return X, Y, x_index, y_index, scaler_x, scaler_y



def make_data(start_date, end_date, symbols, data_dir='data'):

    dates = pd.date_range(start_date, end_date, freq='D')
    df = pd.DataFrame(index=dates)

    for symbol in symbols:
        df_temp = pd.read_csv(get_data_path(symbol, data_dir), index_col="Date", parse_dates=True, na_values=['nan'])
        df_temp = df_temp.reindex(dates)
        
        # 만일 start_date 혹은 end_date 근방 5일의 데이터가 없다면 assert error 발생
        start_date = pd.to_datetime(start_date, format='%Y-%m-%d') + pd.DateOffset(days=4)
        end_date = pd.to_datetime(end_date, format='%Y-%m-%d') - pd.DateOffset(days=4)
        assert not df_temp.loc[:start_date].isnull().all().all(), f'{symbol} has no data before {start_date}.\n Check the data source: {get_data_path(symbol, data_dir)}'
        assert not df_temp.loc[end_date:].isnull().all().all(), f'{symbol} has no data after {end_date}.\n Check the data source: {get_data_path(symbol, data_dir)}'
        
        df_temp.columns = [symbol + '_' + col for col in df_temp.columns]  # rename columns
        df = df.join(df_temp)

    return df



def load_dataset(start_date, end_date, data_dir='data', symbols=SYMBOLS):

    save_dir = os.path.join(data_dir, 'preprocessed')
    os.makedirs(save_dir, exist_ok=True)

    meta_fname = os.path.join(save_dir, 'meta.pkl')
    if not os.path.exists(meta_fname):
        meta = pd.DataFrame(columns=['start', 'end', 'features'])
        meta.index.name = 'fname'
    else:
        meta = pd.read_pickle(meta_fname)

    # symbols가 모두 포함된 경우에 대한 처리
    meta = meta.loc[meta['features'].apply(lambda x: set(symbols).issubset(x))]
    start, end = ''.join(start_date.split('-'))[2:], ''.join(end_date.split('-'))[2:]

    if len(meta[(meta['start'] <= start) & (meta['end'] >= end)]) == 0:
        print(f'Making New Data')
        table = make_data(start_date, end_date, symbols=symbols, data_dir=data_dir)

        for merge_fname, merge_data in meta[((meta['start'] <= start) & (meta['end'] >= start)) | 
                       ((meta['start'] <= end) & (meta['end'] >= end))].iterrows():
            table_tmp = pd.read_pickle(os.path.join(save_dir, merge_fname))
            
            assert (table_tmp[table_tmp.index.isin(table.index)][table.columns].equals(
                table[table.index.isin(table_tmp.index)])), \
                f'Data Mismatch: {merge_fname} {start}-{end}  &  raw data'
            
            table = pd.concat([table, table_tmp[~table_tmp.index.isin(table.index)][table.columns]], axis=0).sort_index()
            start, end = min(start, merge_data['start']), max(end, merge_data['end'])
            
            if table.columns.equals(table_tmp.columns):
                meta = meta.drop(index=merge_fname)
                os.remove(os.path.join(save_dir, merge_fname))
                print(f'Remove Duplicates : {os.path.join(save_dir, merge_fname)}')
        

        for merge_fname in meta[(meta['start'] >= start) & (meta['end'] <= end)].index:
            meta = meta.drop(index=merge_fname)
            os.remove(os.path.join(save_dir, merge_fname))
            print(f'Remove Duplicates : {os.path.join(save_dir, merge_fname)}')


        save_fname = f'{start}_{end}_{str(encode_base64(symbols))[2:12]}.pkl'
        table.to_pickle(os.path.join(save_dir, save_fname))
        table.to_csv(os.path.join(save_dir, save_fname.replace('.pkl', '.csv')))
        meta.loc[save_fname] = [start, end, symbols]
        meta.to_pickle(meta_fname)
        meta.to_csv(meta_fname.replace('.pkl', '.csv'))
        print(f'Save Data to {os.path.join(save_dir, save_fname)}'
              f'\n\tData Period: {start} - {end}'
              f'\n\tSymbols: {symbols}')
        
    else:
        load_data = meta[(meta['start'] <= start) & (meta['end'] >= end)].iloc[0]
        table = pd.read_pickle(os.path.join(save_dir, load_data.name))
        print(f'Load Data from {os.path.join(save_dir, load_data.name)}'
              f'\n\tData Period: {load_data["start"]} - {load_data["end"]}'
              f'\n\tSymbols: {load_data["features"]}')

    return table


if __name__ == "__main__":

    start_date = '2013-01-01'
    end_date = '2023-10-27'

    input_days = 5
    time_lag = 2
    
    is_training = False

    test_data = PriceDataset(start_date, end_date, 
                             phase='all',
                             in_columns=['USD_Price', 'Gold_Price', 'Silver_Price'],
                             out_columns=['USD_Price'],
                             input_days=input_days,
                             time_lag=time_lag,
                             data_dir='data')
    
    print('-'*50)

    for idx in [0, -1]:
        print(f'test_data size : {len(test_data)}')
        print(f'\ndataset_x_index[{idx}] : \n{test_data.x_index[idx]}')
        print(f'\ndataset_x_original[{idx}] : \n{test_data.x[idx]}')
        print(f'\ndataset_y_index[{idx}] : \n{test_data.y_index[idx]}')
        print(f'\ndataset_x_flatten[{idx}] : \n{torch.flatten(test_data.x[idx])}')
        print(f'\ndataset_y[{idx}] : \n{test_data.y[idx]}')
