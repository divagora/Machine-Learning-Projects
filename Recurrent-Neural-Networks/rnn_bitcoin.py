#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 13:52:22 2020

@author: ls616
"""

import numpy as np
import pandas as pd
from pandas import DataFrame

import json
import requests

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg
from pandas.plotting import autocorrelation_plot

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

################
#### IMPORT ####
################

## import btc data ##
endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=1814')
bitcoin = pd.DataFrame(json.loads(res.content)['Data'])
bitcoin = bitcoin.set_index('time')
bitcoin.index = pd.to_datetime(bitcoin.index, unit='s')
bitcoin = bitcoin.drop(["conversionType","conversionSymbol"],axis=1)
#bitcoin["change"] = (bitcoin["close"]-bitcoin["open"])
bitcoin.columns = ["btc_high","btc_low","btc_open","btc_volumefrom","btc_volumeto","btc_close"]#,"btc_change"]
bitcoin.head()

## import eth data ##
endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=ETH&tsym=USD&limit=1814')
eth = pd.DataFrame(json.loads(res.content)['Data'])
eth = eth.set_index('time')
eth.index = pd.to_datetime(eth.index, unit='s')
eth = eth.drop(["conversionType","conversionSymbol"],axis=1)
#eth["change"] = (eth["close"]-eth["open"])
eth.columns = ["eth_high","eth_low","eth_open","eth_volumefrom","eth_volumeto","eth_close"]#,"eth_change"]
eth.head()

## merge data ##
market = pd.concat([bitcoin,eth],axis=1)


################


#######################
#### INITIAL PLOTS ####
#######################

## plot BTC data ##
plt.plot(bitcoin[["btc_close"]],color="C0")
plt.xlabel("Date"); plt.ylabel("BTC (USD)")
plt.show()

## plot ETH data ##
plt.plot(eth[["eth_close"]],color="C0")
plt.xlabel("Date"); plt.ylabel("ETH (USD)")
plt.show()

#######################



##############################
#### TRAINING & TEST DATA ####
##############################

## split data ##
def train_test_split(df, test_size = 0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data


train,test = train_test_split(market,test_size=0.2)

## plot BTC train & test data ##
plt.plot(train["btc_close"],label="Train")
plt.plot(test["btc_close"],label="Test")
plt.ylabel("BTC (USD)"); plt.xlabel("Date")
plt.legend(); plt.show()

## plot ETH train & test data ##
plt.plot(train["eth_close"],label="Train")
plt.plot(test["eth_close"],label="Test")
plt.ylabel("ETH (USD)"); plt.xlabel("Date")
plt.legend(); plt.show()

##############################


#########################################
#### PERSISTENCE (RANDOM WALK) MODEL ####
#########################################

## train data ##
y_btc_train = train["btc_close"][1:]
y_eth_train = train["eth_close"][1:]

## test data ##
y_btc_test = test["btc_close"][1:]
y_eth_test = test["eth_close"][1:]

## 1 step train preds ##
y_hat_btc_train = train["btc_close"].shift(1)[1:]
y_hat_eth_train = train["eth_close"].shift(1)[1:]

## 1 step train errors ##
mse_btc_rw_train = np.sqrt(mean_squared_error(y_btc_train,y_hat_btc_train))
mse_eth_rw_train = np.sqrt(mean_squared_error(y_eth_train,y_hat_eth_train))

## 1 step test preds ##
y_hat_btc_test = test["btc_close"].shift(1)[1:]
y_hat_eth_test = test["eth_close"].shift(1)[1:]

## 1 step test errors ##
mse_btc_rw_test = np.sqrt(mean_squared_error(y_btc_test,y_hat_btc_test))
mse_eth_rw_test = np.sqrt(mean_squared_error(y_eth_test,y_hat_eth_test))

## 1 step test plots ##
plt.plot(y_btc_test,label="True")
plt.plot(y_hat_btc_test,label="Predictions")
plt.ylabel("BTC (USD)"); plt.xlabel("Date")
plt.legend(); plt.show()

plt.plot(y_eth_test,label="True")
plt.plot(y_hat_eth_test,label="Predictions")
plt.ylabel("ETH (USD)"); plt.xlabel("Date")
plt.legend(); plt.show()

## 1 step residuals ##
resid_btc_rw = y_btc_test - y_hat_btc_test
resid_eth_rw = y_eth_test - y_hat_eth_test

## plot 1 step residuals ##
plt.plot(resid_btc_rw,label="BTC")
plt.plot(resid_eth_rw,label="ETH")
plt.ylabel("Residuals"); plt.xlabel("Date")
plt.legend(); plt.show()

## function to extract multi-day output ##
def extract_multi_day_out(df,pred_len):
    window_data_btc = []
    window_data_eth = []
    for i in range(1,len(df)-pred_len):
        tmp_btc = df[i:(i+pred_len)]["btc_close"].copy()
        tmp_eth = df[i:(i+pred_len)]["eth_close"].copy()
        window_data_btc.append(tmp_btc)
        window_data_eth.append(tmp_eth)
    return np.array(window_data_btc), np.array(window_data_eth)

## 5 step targets ##
pred_len = 5
y_btc_test_multi, y_eth_test_multi = extract_multi_day_out(test,pred_len)

## 5-step predictions ##
y_hat_btc_test_multi = np.zeros((y_btc_test_multi.shape[0],5))
for i in range(y_hat_btc_test_multi.shape[0]):
    y_hat_btc_test_multi[i,] = test["btc_close"][i]
  
y_hat_eth_test_multi = np.zeros((y_eth_test_multi.shape[0],5))
for i in range(y_hat_eth_test_multi.shape[0]):
    y_hat_eth_test_multi[i,] = test["eth_close"][i]

## 5 step test errors ##
mse_btc_rw_test_multi = np.sqrt(mean_squared_error(y_btc_test_multi,y_hat_btc_test_multi))
mse_eth_rw_test_multi = np.sqrt(mean_squared_error(y_eth_test_multi,y_hat_eth_test_multi))
    
## 5 hat step predictions for plotting ##
y_hat_btc_test_multi_plot = y_hat_btc_test_multi[::5].flatten()
y_hat_btc_test_multi_plot = pd.Series(y_hat_btc_test_multi_plot,index=test[0:len(y_hat_btc_test_multi_plot)].index)

y_hat_eth_test_multi_plot = y_hat_eth_test_multi[::5].flatten()
y_hat_eth_test_multi_plot = pd.Series(y_hat_eth_test_multi_plot,index=test[0:len(y_hat_eth_test_multi_plot)].index)

## multi-step plots (obviously not good!) ##
plt.plot(y_btc_test,label="True")
for i in range(int(len(y_hat_btc_test_multi_plot)/5)):
    plt.plot(y_hat_btc_test_multi_plot[(5*i):(5*i+5)],label="Predictions")
plt.ylabel("BTC (USD)"); plt.xlabel("Date")
plt.show()

plt.plot(y_eth_test,label="True")
for i in range(int(len(y_hat_eth_test_multi_plot)/5)):
    plt.plot(y_hat_eth_test_multi_plot[(5*i):(5*i+5)],label="Predictions")
plt.ylabel("BTC (USD)"); plt.xlabel("Date")
plt.show()


## multi-step simulations (to do!), need to work out s.d. of errors first ##
## highly dependent on random seed ##

###########################



###################################
#### RANDOM WALK MODEL + DRIFT ####
###################################

## train data ##
y_btc_train = train["btc_close"][1:]
y_eth_train = train["eth_close"][1:]

## test data ##
y_btc_test = test["btc_close"][1:]
y_eth_test = test["eth_close"][1:]

## compute drift ##
btc_drift = np.mean(np.array(y_btc_train[1:])-np.array(y_btc_train[:-1]))
eth_drift = np.mean(np.array(y_eth_train[1:])-np.array(y_eth_train[:-1]))

## 1 step train preds ##
y_hat_btc_train = btc_drift + train["btc_close"].shift(1)[1:]
y_hat_eth_train = eth_drift + train["eth_close"].shift(1)[1:]

## 1 step train errors ##
mse_btc_rw_drift_train = np.sqrt(mean_squared_error(y_btc_train,y_hat_btc_train))
mse_eth_rw_drift_train = np.sqrt(mean_squared_error(y_eth_train,y_hat_eth_train))

## 1 step test preds ##
y_hat_btc_test = btc_drift + test["btc_close"].shift(1)[1:]
y_hat_eth_test = eth_drift + test["eth_close"].shift(1)[1:]

## 1 step error (small improvement from rw with no drift) ##
mse_btc_rw_drift_test = np.sqrt(mean_squared_error(y_btc_test,y_hat_btc_test))
mse_eth_rw_drift_test = np.sqrt(mean_squared_error(y_eth_test,y_hat_eth_test))

## plot ##
plt.plot(y_btc_test,label="True")
plt.plot(y_hat_btc_test,label="Predictions")
plt.ylabel("BTC (USD)"); plt.xlabel("Date")
plt.legend(); plt.show()

plt.plot(y_eth_test,label="True")
plt.plot(y_hat_eth_test,label="Predictions")
plt.ylabel("ETH (USD)"); plt.xlabel("Date")
plt.legend(); plt.show()

## residuals ##
resid_btc_rw_drift = y_btc_test - y_hat_btc_test
resid_eth_rw_drift = y_eth_test - y_hat_eth_test

## plot residuals ##
plt.plot(resid_btc_rw_drift,label="BTC")
plt.plot(resid_eth_rw_drift,label="ETH")
plt.ylabel("Residuals"); plt.xlabel("Date")
plt.legend(); plt.show()


## function to extract multi-day output ##
def extract_multi_day_out(df,pred_len):
    window_data_btc = []
    window_data_eth = []
    for i in range(1,len(df)-pred_len):
        tmp_btc = df[i:(i+pred_len)]["btc_close"].copy()
        tmp_eth = df[i:(i+pred_len)]["eth_close"].copy()
        window_data_btc.append(tmp_btc)
        window_data_eth.append(tmp_eth)
    return np.array(window_data_btc), np.array(window_data_eth)

## 5-step targets ##
pred_len = 5
y_btc_test_multi, y_eth_test_multi = extract_multi_day_out(test,pred_len)

## 5-step predictions ##
y_hat_btc_test_multi = np.zeros((y_btc_test_multi.shape[0],5))
for i in range(y_hat_btc_test_multi.shape[0]):
    for j in range(pred_len):
        y_hat_btc_test_multi[i,j] = test["btc_close"][i]+(j+1)*btc_drift
  
y_hat_eth_test_multi = np.zeros((y_eth_test_multi.shape[0],5))
for i in range(y_hat_eth_test_multi.shape[0]):
    for j in range(pred_len):
        y_hat_eth_test_multi[i,j] = test["eth_close"][i]+(j+1)*eth_drift

## 5-step test errors ##
mse_btc_rw_drift_test_multi = np.sqrt(mean_squared_error(y_btc_test_multi,y_hat_btc_test_multi))
mse_eth_rw_drift_test_multi = np.sqrt(mean_squared_error(y_eth_test_multi,y_hat_eth_test_multi))
    
## 5- step predictions for plotting ##
y_hat_btc_test_multi_plot = y_hat_btc_test_multi[::5].flatten()
y_hat_btc_test_multi_plot = pd.Series(y_hat_btc_test_multi_plot,index=test[0:len(y_hat_btc_test_multi_plot)].index)

y_hat_eth_test_multi_plot = y_hat_eth_test_multi[::5].flatten()
y_hat_eth_test_multi_plot = pd.Series(y_hat_eth_test_multi_plot,index=test[0:len(y_hat_eth_test_multi_plot)].index)

## 5-step plots (obviously not good!) ##
plt.plot(y_btc_test,label="True")
for i in range(int(len(y_hat_btc_test_multi_plot)/5)):
    plt.plot(y_hat_btc_test_multi_plot[(5*i):(5*i+5)],label="Predictions")
plt.ylabel("BTC (USD)"); plt.xlabel("Date")
plt.show()

plt.plot(y_eth_test,label="True")
for i in range(int(len(y_hat_eth_test_multi_plot)/5)):
    plt.plot(y_hat_eth_test_multi_plot[(5*i):(5*i+5)],label="Predictions")
plt.ylabel("ETH (USD)"); plt.xlabel("Date")
plt.show()

## multi-step simulations (to do!), need to work out s.d. of errors first ##
## highly dependent on random seed ##

################################################



######################
#### ARIMA MODELS ####
######################

## https://otexts.com/fpp2/arima.html

## train data ##
y_btc_train = train["btc_close"]
y_eth_train = train["eth_close"]

## function to test stationarity ##
def test_stationarity(x):

    #Determing rolling statistics
    rolmean = x.rolling(window=22,center=False).mean()

    rolstd = x.rolling(window=12,center=False).std()
    
    #Plot rolling statistics:
    orig = plt.plot(x, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey Fuller test    
    result=adfuller(x)
    print('ADF Stastistic: %f'%result[0])
    print('p-value: %f'%result[1])
    pvalue=result[1]
    for key,value in result[4].items():
         if result[0]>value:
            print("The data is non-stationary")
            break
         else:
            print("The data is stationary")
            break;
    print('Critical values:')
    for key,value in result[4].items():
        print('\t%s: %.3f ' % (key, value))
   
## test stationarity: original data not stationary, so...         
test_stationarity(y_btc_train)
test_stationarity(y_eth_train)

## difference data ##
y_btc_train_diff = y_btc_train[1:] - y_btc_train.shift()[1:]
y_eth_train_diff = y_eth_train[1:] - y_eth_train.shift()[1:]

## test stationarity: now stationary, so can fit ARIMA models with d = 1 ...
test_stationarity(y_btc_train_diff)
test_stationarity(y_eth_train_diff) 


### ARIMA(0,1,0) model (this is equivalent to RW + DRIFT) ###

## train data ##
y_btc_train = train["btc_close"]
y_eth_train = train["eth_close"]

## test data ##
y_btc_test = test["btc_close"]
y_eth_test = test["eth_close"]

## 1 step test preds ##
y_hat_btc_test = np.zeros(len(y_btc_test))
y_hat_eth_test = np.zeros(len(y_eth_test))

for i in range(0,len(y_btc_test)):
    mod_btc = ARIMA(y_btc_train,order=(0,1,0))
    mod_eth = ARIMA(y_eth_train,order=(0,1,0))
    
    mod_btc_fit = mod_btc.fit()
    mod_eth_fit = mod_eth.fit()
    
    out_btc = mod_btc_fit.forecast()
    out_eth = mod_eth_fit.forecast()
    
    pred_btc = out_btc[0]
    pred_eth = out_eth[0]
    
    y_hat_btc_test[i] = pred_btc
    y_hat_eth_test[i] = pred_eth
    
    y_btc_train = np.append(y_btc_train,y_btc_test[i])
    y_eth_train = np.append(y_eth_train,y_eth_test[i])
    
y_hat_btc_test = pd.Series(y_hat_btc_test,index=y_btc_test.index)
y_hat_eth_test = pd.Series(y_hat_eth_test,index=y_eth_test.index)

## 1 step test errors ##
mse_btc_arima010 = np.sqrt(mean_squared_error(y_btc_test,y_hat_btc_test))
mse_eth_arima010 = np.sqrt(mean_squared_error(y_eth_test,y_hat_eth_test))

## slightly better than rw_drift since now include test data when fitting ##

## plots ##
plt.plot(y_btc_test,label="True")
plt.plot(y_hat_btc_test,label="Predictions")
plt.ylabel("BTC (USD)"); plt.xlabel("Date")
plt.legend(); plt.show()

plt.plot(y_eth_test,label="True")
plt.plot(y_hat_eth_test,label="Predictions")
plt.ylabel("ETH (USD)"); plt.xlabel("Date")
plt.legend(); plt.show()

## residuals ##
resid_btc_arima010 = y_btc_test - y_hat_btc_test
resid_eth_arima010 = y_eth_test - y_hat_eth_test

## plot residuals ##
plt.plot(resid_btc_arima010,label="BTC")
plt.ylabel("Residuals");plt.xlabel("Date")
plt.legend(); plt.show()

plt.plot(resid_eth_arima010,label="BTC")
plt.ylabel("Residuals");plt.xlabel("Date")
plt.legend(); plt.show()


### Analyse Correlations ###

## check acf (train data): significant correlations at 5, 10, 17, 19, 20, 32, 33
plot_acf(y_btc_train_diff,lags=50)

## check acf (test data): significant correlations at 12, 20 
plot_acf(y_btc_test[1:] - y_btc_test.shift()[1:],lags=50)



### ARIMA(2,1,0) model ###

## train data ##
y_btc_train = train["btc_close"]
y_eth_train = train["eth_close"]

## test data ##
y_btc_test = test["btc_close"]
y_eth_test = test["eth_close"]

## 1 step test preds ##
y_hat_btc_test = np.zeros(len(y_btc_test))
y_hat_eth_test = np.zeros(len(y_eth_test))

pred_len = 1

for i in range(0,len(y_btc_test)):
    mod_btc = ARIMA(y_btc_train,order=(2,1,0))
    mod_eth = ARIMA(y_eth_train,order=(2,1,0))
    
    mod_btc_fit = mod_btc.fit()
    mod_eth_fit = mod_eth.fit()
    
    out_btc = mod_btc_fit.forecast(steps=pred_len)
    out_eth = mod_eth_fit.forecast(steps=pred_len)
    
    pred_btc = out_btc[0]
    pred_eth = out_eth[0]
    
    y_hat_btc_test[i] = pred_btc
    y_hat_eth_test[i] = pred_eth
    
    y_btc_train = np.append(y_btc_train,y_btc_test[i])
    y_eth_train = np.append(y_eth_train,y_eth_test[i])

    
y_hat_btc_test = pd.Series(y_hat_btc_test,index=y_btc_test.index)
y_hat_eth_test = pd.Series(y_hat_eth_test,index=y_eth_test.index)

## 1 step test errors (worse than AR(0,1,0)) ##
mse_btc_arima510 = np.sqrt(mean_squared_error(y_btc_test,y_hat_btc_test))
mse_eth_arima510 = np.sqrt(mean_squared_error(y_eth_test,y_hat_eth_test))

## plots ##
plt.plot(y_btc_test,label="True")
plt.plot(y_hat_btc_test,label="Predictions")
plt.ylabel("BTC (USD)"); plt.xlabel("Date")
plt.legend(); plt.show()

plt.plot(y_eth_test,label="True")
plt.plot(y_hat_eth_test,label="Predictions")
plt.ylabel("ETH (USD)"); plt.xlabel("Date")
plt.legend(); plt.show()

## residuals ##
resid_btc_arima510 = y_btc_test - y_hat_btc_test
resid_eth_arima510 = y_eth_test - y_hat_eth_test

## plot residuals ##
plt.plot(resid_btc_arima510,label="BTC")
plt.ylabel("Residuals");plt.xlabel("Date")
plt.legend(); plt.show()

plt.plot(resid_eth_arima510,label="BTC")
plt.ylabel("Residuals");plt.xlabel("Date")
plt.legend(); plt.show()


## 5-step targets ##
pred_len = 5
y_btc_test_multi, y_eth_test_multi = extract_multi_day_out(test,pred_len)

## 5-step preds ##
y_hat_btc_test_multi = np.zeros((y_btc_test_multi.shape[0],5))
y_hat_eth_test_multi = np.zeros((y_eth_test_multi.shape[0],5))


for i in range(y_btc_test_multi.shape[0]):
    
    y_btc_train = np.append(y_btc_train,y_btc_test[i])
    y_eth_train = np.append(y_eth_train,y_eth_test[i])
    
    mod_btc = ARIMA(y_btc_train,order=(5,1,0))
    mod_eth = ARIMA(y_eth_train,order=(5,1,0))
    
    mod_btc_fit = mod_btc.fit()
    mod_eth_fit = mod_eth.fit()
    
    out_btc = mod_btc_fit.forecast(steps=pred_len)
    out_eth = mod_eth_fit.forecast(steps=pred_len)
    
    pred_btc = np.array(out_btc)
    pred_eth = np.array(out_eth)
    
    y_hat_btc_test_multi[i,] = pred_btc
    y_hat_eth_test_multi[i,] = pred_eth
    
    
## 5-step test errors (worse than rw!) ##
mse_btc_arima510_test_multi = np.sqrt(mean_squared_error(y_btc_test_multi,y_hat_btc_test_multi))
mse_eth_atima510_test_multi = np.sqrt(mean_squared_error(y_eth_test_multi,y_hat_eth_test_multi))
    
## 5- step predictions for plotting ##
y_hat_btc_test_multi_plot = y_hat_btc_test_multi[::5].flatten()
y_hat_btc_test_multi_plot = pd.Series(y_hat_btc_test_multi_plot,index=test[0:len(y_hat_btc_test_multi_plot)].index)

y_hat_eth_test_multi_plot = y_hat_eth_test_multi[::5].flatten()
y_hat_eth_test_multi_plot = pd.Series(y_hat_eth_test_multi_plot,index=test[0:len(y_hat_eth_test_multi_plot)].index)

## 5-step plots (obviously not good!) ##
plt.plot(y_btc_test,label="True")
for i in range(int(len(y_hat_btc_test_multi_plot)/5)):
    plt.plot(y_hat_btc_test_multi_plot[(5*i):(5*i+5)],label="Predictions")
plt.ylabel("BTC (USD)"); plt.xlabel("Date")
plt.show()

plt.plot(y_eth_test,label="True")
for i in range(int(len(y_hat_eth_test_multi_plot)/5)):
    plt.plot(y_hat_eth_test_multi_plot[(5*i):(5*i+5)],label="Predictions")
plt.ylabel("ETH (USD)"); plt.xlabel("Date")
plt.show()

######################



##################################
#### RNN (LSTM) MODEL (1-day) ####
##################################

## function to normalise window data  (to 1st value in window) ##
def normalise_window(df):
    return df / df.iloc[0] - 1

## function to extract window data ##
def extract_window(df, window_len=10, normalise=True):
    window_data = []
    for i in range(len(df) - window_len):
        tmp = df[i: (i + window_len)].copy()
        if normalise:
            tmp = normalise_window(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)

## function to prepare all data ##
def prepare_data(df, window_len=10, normalise=True, test_size=0.2):
    
    # train test split
    train_data, test_data = train_test_split(df, test_size=test_size)
    
    # extract window data
    X_train = extract_window(train_data, window_len, normalise)
    X_test = extract_window(test_data, window_len, normalise)
    
    # extract targets
    y_btc_train = train_data["btc_close"][window_len:].values
    y_btc_test = test_data["btc_close"][window_len:].values
    
    y_eth_train = train_data["eth_close"][window_len:].values
    y_eth_test = test_data["eth_close"][window_len:].values
    
    if normalise:
        y_btc_train = y_btc_train / train_data["btc_close"][:-window_len].values - 1
        y_btc_test = y_btc_test / test_data["btc_close"][:-window_len].values - 1
        
        y_eth_train = y_eth_train / train_data["eth_close"][:-window_len].values - 1
        y_eth_test = y_eth_test / test_data["eth_close"][:-window_len].values - 1

    return train_data, test_data, X_train, X_test, y_btc_train, y_btc_test, y_eth_train, y_eth_test


## data params ##
window_len = 2; normalise = True; test_size = 0.2

## data ##
train,test,x_train,x_test,y_btc_train,y_btc_test,y_eth_train,y_eth_test = prepare_data(market,window_len,normalise,test_size)

## function to build model ##
def build_model(input_data,output_size,neurons=20,activ_func="linear",
                dropout=0.25,loss="mae",optimizer="adam"):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

## model params ##
neurons = 20; output_size = 1; 
dropout = 0.25; epochs = 100; 
batch_size = 50; verbose = 1; 
shuffle = True

## model architecture ##
np.random.seed(1)
lstm_btc_model = build_model(x_train,output_size=output_size,neurons=neurons)
lstm_eth_model = build_model(x_train,output_size=output_size,neurons=neurons)

## fit model ##
np.random.seed(1)
btc_history = lstm_btc_model.fit(x_train,y_btc_train,epochs=epochs,batch_size=batch_size,verbose=verbose,shuffle=shuffle,validation_data = (x_test,y_btc_test))
eth_history = lstm_eth_model.fit(x_train,y_eth_train,epochs=epochs,batch_size=batch_size,verbose=verbose,shuffle=shuffle,validation_data = (x_test,y_eth_test))

## plot training & test error ##
plt.plot(btc_history.epoch,btc_history.history["loss"])
plt.plot(btc_history.epoch,btc_history.history["val_loss"])
plt.title("Training Loss");
plt.xlabel("Epochs"); plt.ylabel("MAE")
plt.show()

plt.plot(eth_history.epoch,eth_history.history["loss"])
plt.plot(eth_history.epoch,eth_history.history["val_loss"])
plt.title("Training Loss")
plt.xlabel("Epochs"); plt.ylabel("MAE")
plt.show()

## train targets ##
y_target_btc_train = train["btc_close"][window_len:]
y_target_eth_train = train["eth_close"][window_len:]

## train preds ##
y_hat_btc_train = pd.Series(((np.transpose(lstm_btc_model.predict(x_train))+1) * train['btc_close'].values[:-window_len])[0],index=y_target_btc_train.index)
y_hat_eth_train = pd.Series(((np.transpose(lstm_eth_model.predict(x_train))+1) * train['eth_close'].values[:-window_len])[0],index=y_target_eth_train.index)

## train error ##
mse_btc_lstm_train = np.sqrt(mean_squared_error(y_target_btc_train,y_hat_btc_train))
mse_eth_lstm_train = np.sqrt(mean_squared_error(y_target_eth_train,y_hat_eth_train))

## plot train preds ##
plt.plot(y_target_btc_train, label='Actual')
plt.plot(y_hat_btc_train,label="Predicted")
plt.legend(); plt.show()

plt.plot(y_target_eth_train, label='Actual')
plt.plot(y_hat_eth_train,label="Predicted")
plt.legend(); plt.show()


## test targets ##
y_target_btc_test = test["btc_close"][window_len:]
y_target_eth_test = test["eth_close"][window_len:]

## test preds ##
y_hat_btc_test = pd.Series(((np.transpose(lstm_btc_model.predict(x_test))+1) * test['btc_close'].values[:-window_len])[0],index=y_target_btc_test.index)
y_hat_eth_test = pd.Series(((np.transpose(lstm_eth_model.predict(x_test))+1) * test['eth_close'].values[:-window_len])[0],index=y_target_eth_test.index)

## test error ##
mse_btc_lstm_test = np.sqrt(mean_squared_error(y_target_btc_test,y_hat_btc_test))
mse_eth_lstm_test = np.sqrt(mean_squared_error(y_target_eth_test,y_hat_eth_test))

## plot train preds ##
plt.plot(y_target_btc_test, label='True')
plt.plot(y_hat_btc_test,label="Predictions")
plt.legend(); plt.show()

plt.plot(y_target_eth_test, label='True')
plt.plot(y_hat_eth_test,label="Predictions")
plt.legend(); plt.show()

## residuals ##
resid_btc_lstm = y_target_btc_test - y_hat_btc_test
resid_eth_lstm = y_target_eth_test - y_hat_eth_test

## plot residuals ##
plt.plot(resid_btc_lstm,label="BTC")
plt.ylabel("Residuals");plt.xlabel("Date")
plt.legend(); plt.show()

plt.plot(resid_eth_lstm,label="ETH")
plt.ylabel("Residuals");plt.xlabel("Date")
plt.legend(); plt.show()


##################################


##################################
#### RNN (LSTM) MODEL (multi-day) ####
##################################

## function to normalise window data  (to 1st value in window) ##
def normalise_window(df):
    return df / df.iloc[0] - 1

## function to extract window data ##
def extract_window(df, window_len=10, normalise=True):
    window_data = []
    for i in range(len(df) - window_len):
        tmp = df[i: (i + window_len)].copy()
        if normalise:
            tmp = normalise_window(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)

## function to extract window output ##
def extract_out_window(df,window_len,pred_len,normalise=True):
    window_data_btc = []
    window_data_eth = []
    for i in range(window_len,len(df)-pred_len):
        tmp_btc = df[i:(i+pred_len)]["btc_close"].copy()
        tmp_eth = df[i:(i+pred_len)]["eth_close"].copy()
        if normalise:
            tmp_btc = tmp_btc/df[(i-window_len):(i-window_len+1)]["btc_close"].values - 1
            tmp_eth = tmp_eth/df[(i-window_len):(i-window_len+1)]["eth_close"].values - 1
        window_data_btc.append(tmp_btc)
        window_data_eth.append(tmp_eth)
    return np.array(window_data_btc), np.array(window_data_eth)
    
## function to prepare all data ##
def prepare_data(df, window_len=10, pred_len = 1, normalise=True, test_size=0.2):
    
    # train test split
    train_data, test_data = train_test_split(df, test_size=test_size)
    
    # extract window data
    X_train = extract_window(train_data, window_len, normalise)[:-pred_len]
    X_test = extract_window(test_data, window_len, normalise)[:-pred_len]
    
    # extract window targets
    y_btc_train, y_eth_train = extract_out_window(train_data,window_len,pred_len,normalise)
    y_btc_test, y_eth_test = extract_out_window(test_data,window_len,pred_len,normalise)

    return train_data, test_data, X_train, X_test, y_btc_train, y_btc_test, y_eth_train, y_eth_test

## data params ##
window_len = 2; pred_len = 5; normalise = True; test_size = 0.2

## data ##
train,test,x_train,x_test,y_btc_train,y_btc_test,y_eth_train,y_eth_test = prepare_data(market,window_len,normalise,test_size)

## function to build model ##
def build_model(input_data,output_size,neurons=20,activ_func="linear",
                dropout=0.25,loss="mae",optimizer="adam"):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

## model params ##
neurons = 20; output_size = pred_len; 
dropout = 0.25; epochs = 100; 
batch_size = 50; verbose = 1; 
shuffle = True

## model architecture ##
np.random.seed(1)
lstm_btc_model = build_model(x_train,output_size=output_size,neurons=neurons)
lstm_eth_model = build_model(x_train,output_size=output_size,neurons=neurons)

## fit model ##
np.random.seed(1)
btc_history = lstm_btc_model.fit(x_train,y_btc_train,epochs=epochs,batch_size=batch_size,verbose=verbose,shuffle=shuffle,validation_data = (x_test,y_btc_test))
eth_history = lstm_eth_model.fit(x_train,y_eth_train,epochs=epochs,batch_size=batch_size,verbose=verbose,shuffle=shuffle,validation_data = (x_test,y_eth_test))

## plot training & test error ##
plt.plot(btc_history.epoch,btc_history.history["loss"])
plt.plot(btc_history.epoch,btc_history.history["val_loss"])
plt.title("Training Loss");
plt.xlabel("Epochs"); plt.ylabel("MAE")
plt.show()

plt.plot(eth_history.epoch,eth_history.history["loss"])
plt.plot(eth_history.epoch,eth_history.history["val_loss"])
plt.title("Training Loss")
plt.xlabel("Epochs"); plt.ylabel("MAE")
plt.show()

## train targets ##
y_target_btc_train = train["btc_close"][window_len:(int(np.ceil((len(x_train)-pred_len)/float(pred_len)))*pred_len+window_len)]
y_target_eth_train = train["eth_close"][window_len:(int(np.ceil((len(x_train)-pred_len)/float(pred_len)))*pred_len+window_len)]

## train preds ##
y_hat_btc_train = pd.Series(((lstm_btc_model.predict(x_train)[:-pred_len][::pred_len]+1)*train['btc_close'].values[:-(window_len + pred_len)][::pred_len].reshape(int(np.ceil((len(x_train)-pred_len)/float(pred_len))),1)).reshape(len(y_target_btc_train)),index=y_target_btc_train.index)
y_hat_eth_train = pd.Series(((lstm_eth_model.predict(x_train)[:-pred_len][::pred_len]+1)*train['eth_close'].values[:-(window_len + pred_len)][::pred_len].reshape(int(np.ceil((len(x_train)-pred_len)/float(pred_len))),1)).reshape(len(y_target_eth_train)),index=y_target_eth_train.index)

## train error ##
mse_btc_lstm5d_train = np.sqrt(mean_squared_error(y_target_btc_train,y_hat_btc_train))
mse_eth_lstm5d_train = np.sqrt(mean_squared_error(y_target_eth_train,y_hat_eth_train))

## plot train preds ##
plt.plot(y_target_btc_train, label='Actual')
plt.plot(y_hat_btc_train,label="Predicted")
plt.legend(); plt.show()

plt.plot(y_target_eth_train, label='Actual')
plt.plot(y_hat_eth_train,label="Predicted")
plt.legend(); plt.show()


## test targets ##
y_target_btc_test = test["btc_close"][window_len:(int(np.ceil((len(x_test)-pred_len)/float(pred_len)))*pred_len+window_len)]
y_target_eth_test = test["eth_close"][window_len:(int(np.ceil((len(x_test)-pred_len)/float(pred_len)))*pred_len+window_len)]

## test preds ##
y_hat_btc_test = pd.Series(((lstm_btc_model.predict(x_test)[:-pred_len][::pred_len]+1)*test['btc_close'].values[window_len:(int(np.ceil((len(x_test)-pred_len)/float(pred_len)))*pred_len+window_len)][::pred_len].reshape(int(np.ceil((len(x_test)-pred_len)/float(pred_len))),1)).reshape(len(y_target_btc_test)),index=y_target_btc_test.index)
y_hat_eth_test = pd.Series(((lstm_eth_model.predict(x_test)[:-pred_len][::pred_len]+1)*test['eth_close'].values[window_len:(int(np.ceil((len(x_test)-pred_len)/float(pred_len)))*pred_len+window_len)][::pred_len].reshape(int(np.ceil((len(x_test)-pred_len)/float(pred_len))),1)).reshape(len(y_target_eth_test)),index=y_target_eth_test.index)

## test error ##
mse_btc_lstm_test_multi = np.sqrt(mean_squared_error(y_target_btc_test,y_hat_btc_test))
mse_eth_lstm_test_multi = np.sqrt(mean_squared_error(y_target_eth_test,y_hat_eth_test))

## plot train preds ##
plt.plot(y_target_btc_test, label='True')
for i in range((int(np.ceil((len(x_test)-pred_len)/float(pred_len))))):
    plt.plot(y_hat_btc_test[pred_len*i:pred_len*i+5],label="Predictions")
plt.show()

plt.plot(y_target_eth_test, label='True')
for i in range((int(np.ceil((len(x_test)-pred_len)/float(pred_len))))):
    plt.plot(y_hat_eth_test[pred_len*i:pred_len*i+5],label="Predictions")
plt.show()

## residuals ##
resid_btc_lstm_multi = y_target_btc_test - y_hat_btc_test
resid_eth_lstm_multi = y_target_eth_test - y_hat_eth_test

## plot residuals ##
plt.plot(resid_btc_lstm,label="BTC")
plt.ylabel("Residuals");plt.xlabel("Date")
plt.legend(); plt.show()

plt.plot(resid_eth_lstm,label="ETH")
plt.ylabel("Residuals");plt.xlabel("Date")
plt.legend(); plt.show()


##################################


