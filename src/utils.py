# utils.py

import os
import random
import numpy as np
import pandas as pd
import torch

import base64
import time
import logging
from omegaconf import OmegaConf
from utils_vis import *


METRIC_NAMES = {
    'RMSELoss': 'RMSE',
    'MSELoss': 'MSE',
    'MAELoss': 'MAE',
    'L1Loss': 'MAE',
    'MAPE': 'MAPE',
}


def get_data_path(symbol, data_dir='data'):

    commodity_dir = os.path.join(data_dir, 'commodities')
    currency_dir = os.path.join(data_dir, 'currencies')

    if symbol in ['AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY', 'USD']:
        path = os.path.join(currency_dir, symbol + '.csv')
    elif symbol in ['BrentOil', 'Copper', 'CrudeOil', 'Gasoline', 'Gold', 'NaturalGas', 'Platinum', 'Silver']:
        path = os.path.join(commodity_dir, symbol + '.csv')
    elif symbol in ['test']:
        path = os.path.join(data_dir, 'test.csv')
    else:
        raise ValueError(f'Invalid symbol: {symbol}')

    return path


def encode_base64(lst: list) -> str:
    return base64.b64encode(','.join(lst).encode())

def decode_base64(s: str) -> list:
    return base64.b64decode(s).decode().split(',')


class Setting:
    @staticmethod
    def seed_everything(seed):
        '''
        [description]
        seed 값을 고정시키는 함수

        [arguments]
        seed : seed 값
        '''
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def __init__(self):
        now = time.localtime()
        now_date = time.strftime('%Y%m%d', now)
        now_hour = time.strftime('%X', now)
        save_time = now_date + '_' + now_hour.replace(':', '')
        self.save_time = save_time

    def get_log_path(self, args):
        '''
        [description]
        log file을 저장할 경로를 반환하는 함수

        [arguments]
        args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.

        [return]
        path : log file을 저장할 경로를 반환합니다.
        이 때, 경로는 saved/log/날짜_시간_모델명/ 입니다.
        '''
        path = os.path.join(args.save_dir.log, f'{self.save_time}_{args.model}/')
        os.makedirs(path, exist_ok=True)
        
        return path

    def get_result_filename(self, args):
        '''
        [description]
        result file을 저장할 경로를 반환하는 함수입니다.

        [arguments]
        args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.

        [return]
        filename : result file을 저장할 경로를 반환합니다.
        이 때, 파일명은 saved/result/날짜_시간_모델명.xlsx 입니다.
        '''
        filename = os.path.join(args.save_dir.result, f'{self.save_time}_{args.model}.xlsx')
            
        return filename


class Logger:
    def __init__(self, args, path):
        """
        [description]
        log file을 생성하는 클래스

        [arguments]
        args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.
        path : log file을 저장할 경로를 전달받습니다.
        """
        self.args = args
        self.path = path

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('[%(asctime)s] - %(message)s')

        self.file_handler = logging.FileHandler(os.path.join(self.path, 'train.log'))
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def log(self, args, epoch: int, verbose: bool = True,
            message: str = '',
            train_loss: float = None, train_errs: dict = None, 
            valid_loss: float = None, valid_errs: dict = None,
            test_errs: dict = None):
        '''
        [description]
        log file에 epoch, train loss, valid loss를 기록하는 함수
        이 때, log file은 train.log로 저장됩니다.

        [arguments]
        epoch : epoch
        train_loss : train loss
        valid_loss : valid loss
        '''
        if epoch > 0:
            message = [f'[Epoch {epoch:02d}] ' + message]
        else:
            message = [message]

        if train_loss:
            message.append(f'TRAIN LOSS({METRIC_NAMES[args.loss]}) : {train_loss:.3f}')
        
        if train_errs:
            for metric, value in train_errs.items():
                message.append(f'TRAIN {metric} : {value:.3f}')
        if valid_loss:
            message.append(f'VALID LOSS({METRIC_NAMES[args.loss]}) : {valid_loss:.3f}')
        if valid_errs:
            for metric, value in valid_errs.items():
                message.append(f'VALID {metric} : {value:.3f}')
        if test_errs:
            for metric, value in test_errs.items():
                message.append(f'TEST {metric} : {value:.3f}')

        message = message[0] + ' | '.join(message[1:])
        self.logger.info(message)
        if verbose:
            print(message)


    def close(self):
        '''
        [description]
        log file을 닫는 함수
        '''
        self.logger.removeHandler(self.file_handler)
        self.file_handler.close()

    def save_args(self):
        '''
        [description]
        model에 사용된 args를 저장하는 함수
        이 때, 저장되는 파일명은 model.json으로 저장됩니다.
        '''
        with open(os.path.join(self.path, 'config.yaml'), 'w') as f:
            OmegaConf.save(self.args, f)

    def __del__(self):
        self.close()


class MetricTracker:
    def __init__(self, *keys):
        self._data = {
            key: pd.DataFrame(columns=['total', 'counts', 'average'])
            for key in keys
        }
        self.reset()

    def reset(self):
        for key in self._data.keys():
            self._data[key] = pd.DataFrame(columns=['total', 'counts', 'average'])
            self._data[key].loc[1] = [0., 0, 0.]

    def update(self, key, value, n=1, epoch=-1):
        if epoch == -1:
            epoch = self._data[key].index[-1]
        if epoch not in self._data[key].index:
            self._data[key].loc[epoch] = [0., 0, 0.]
        self._data[key].loc[epoch, 'total'] += value * n
        self._data[key].loc[epoch, 'counts'] += n
        self._data[key].loc[epoch, 'average'] =self._data[key].loc[epoch, 'total'] / self._data[key].loc[epoch, 'counts']

    def value(self, key, epoch=-1):
        if epoch == -1:
            epoch = self._data[key].index[-1]
        return self._data[key].loc[epoch, 'average']

    def result(self):
        return pd.DataFrame({
            key: self._data[key].loc[:, 'average'] for key in self._data.keys()},
        )

    def is_best(self, key, epoch=-1, mode='min'):
        if epoch == -1:
            epoch = self._data[key].index[-1]
        if mode == 'min':
            return self.value(key, epoch) == self._data[key].loc[:, 'average'].min()
        elif mode == 'max':
            return self.value(key, epoch) == self._data[key].loc[:, 'average'].max()
    



