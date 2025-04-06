import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from typing import List, Union, Optional
from numpy.lib.stride_tricks import sliding_window_view

class Yahoo_Downloader:
    
    def __init__(self, tickers:List[str], date_start: str="2010-07-01", date_end="2019-07-01", stride:int=1):
        self.tickers = tickers
        self.date_start = date_start
        self.date_end = date_end
        self.n_assets = len(tickers)
        self.stride = 1
        self.pd_data = yf.download(tickers=self.tickers, start=date_start, end=date_end)
        self.raw_data = self.pd_data.values
        self.n_samples = self.raw_data.shape[0]
        self.n_features = self.raw_data.shape[1]
        
        if self.pd_data.isnull().sum().sum() > 0:
            raise ValueError('Utilisez des actions sans valeurs manquantes ou bien choisissez une autre période')
        
    def _compute_data_training(self, N:int=5):
        self.first_index_window = N
        X = sliding_window_view(self.raw_data, (N, self.raw_data.shape[1])).reshape(-1, N, self.raw_data.shape[1])
        close_data = self.pd_data['Close'].values
        y = (close_data[N:] > close_data[N-1:-1]).astype(int)
        
        return X[:-1], y

    
    def compute_train_val_test(self, N:int=5, tensor:bool=False):
        
        X, y = self._compute_data_training(N)
        X_torch = torch.from_numpy(X).float()
        
        mean = X_torch.mean(dim=(0,1))
        std  = X_torch.std(dim=(0,1), unbiased=False)
        
        std[std < 1e-8] = 1e-8
        X_torch = (X_torch - mean[None, None, :]) / std[None, None, :]
        
    
        train_rate = 0.8
        test_rate, val_rate = 0.1, 0.1
        
        T = X.shape[0]
        
        self.train_periods = (0, int(np.floor(train_rate*T)))
        self.val_periods = (int(np.floor(train_rate*T)), int(np.floor(train_rate*T))+int(np.floor(val_rate*T)))
        self.test_periods = (int(np.floor(train_rate*T))+int(np.floor(val_rate*T)), int(np.floor(train_rate*T))+2*int(np.floor(val_rate*T)))
        
        Xtrain, ytrain = X[self.train_periods[0]:self.train_periods[1], :, :], y[self.train_periods[0]:self.train_periods[1]]
        Xval, yval = X[self.val_periods[0]:self.val_periods[1], :, :], y[self.val_periods[0]:self.val_periods[1]]
        Xtest, ytest = X[self.test_periods[0]:self.test_periods[1], :, :], y[self.test_periods[0]:self.test_periods[1]]
        
        if not torch:
            return Xtrain.numpy(), ytrain.numpy(), Xtest.numpy(), ytest.numpy(), Xval.numpy(), yval.numpy()
        
        return Xtrain, ytrain, Xval, yval, Xtest, ytest
    
    
        
        

        
        
        
        
