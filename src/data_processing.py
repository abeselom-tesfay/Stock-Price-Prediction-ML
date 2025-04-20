import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def download_stock_data(stock='GOOG', start='2015-01-01', end='2025-01-01'):
    data = yf.download(stock, start=start, end=end, auto_adjust=True)
    data.reset_index(inplace=True)
    return data

def split_data(data, train_ratio=0.8):
    train_size = int(len(data) * train_ratio)
    data_train = pd.DataFrame(data.Close[:train_size])
    data_test = pd.DataFrame(data.Close[train_size:])
    return data_train, data_test

def scale_data(data_train):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_train)
    return scaled_data, scaler

def prepare_sequences(scaled_data, time_step=100):
    x, y = [], []
    for i in range(time_step, len(scaled_data)):
        x.append(scaled_data[i-time_step:i])
        y.append(scaled_data[i, 0])
    return np.array(x), np.array(y)
