# Importing functions and classes we'll use

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dropout, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from warnings import catch_warnings
from warnings import filterwarnings
import keras
import sys
import scipy.stats
import json
import numpy.fft
import time
from decimal import Decimal
import math
from math import sqrt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler

from .ASAP import *
from .esn import *

def sarima_multistep_forecast(history, config, window_size, n_steps):
    order, sorder, trend = config
    new_hist = history[:]
    yhat = []
    # define model
    for i in range(n_steps):

      model = SARIMAX(new_hist[-window_size:], order=order, seasonal_order=sorder, trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)

      # fit model
      model_fit = model.fit(disp=False)

      # make multistep forecast
      # yhat = model_fit.forecast(steps=n_steps)

      prediction = model_fit.predict(start=len(history[-window_size:]), end=len(history[-window_size:]))

      yhat = np.append(yhat, prediction)
      new_hist = np.append(new_hist, prediction)
      new_hist = new_hist[1:]
    return yhat


def auto_forecast(models, input_data, horizon, hyperparameters):
    
    if hyperparameters["forecasting_model"] in ["RNN", "LSTM", "GRU"]:

        x = np.reshape(input_data, (1, 1, hyperparameters['look_back']))
        predictions = models.predict(np.array(x))
        return predictions
    
    elif hyperparameters["forecasting_model"] == "ESN":

        x = np.reshape(input_data, (1, hyperparameters['look_back']))
        
        # if horizon > 1:
        #     def esn_recursive_strategy(model, X_row, n_steps):
        #         forecasts = []

        #         for i in range(n_steps):
        #             print(X_row.shape)
        #             forecast = model.predict(np.array(X_row))
        #             forecasts.append(forecast[0, 0])
        #             X_row = X_row.tolist()
        #             X_row.append(forecast[0, 0])
        #             X_row = X_row[1:]
        #             X_row = np.array(X_row)
        #         return forecasts
        #     def esn_make_predictions(model, X, n_steps):
        #         predictions = []
        #         row_forecasts = esn_recursive_strategy(model, X[:], n_steps)
        #         predictions.append(row_forecasts)
        #         return predictions
            
        #     predictions = esn_make_predictions(models, x, horizon)

        if horizon > 1:
            def esn_recursive_strategy(model, X_row, n_steps):
                forecasts = []
                X_row = np.array(X_row) 
                
                for i in range(n_steps):
                    # Ensure X_row is reshaped to (1, num_features) for prediction
                    forecast = model.predict(X_row.reshape(1, -1))
                    forecasts.append(forecast[0, 0])
                    
                    # Update X_row by appending the new forecast and removing the oldest element
                    X_row = np.append(X_row[:, 1:], forecast[0, 0].reshape(1, 1), axis=1)
                    
                return forecasts

            def esn_make_predictions(model, X, n_steps):
                predictions = []
                row_forecasts = esn_recursive_strategy(model, np.array(X), n_steps)
                predictions.append(row_forecasts)
                return predictions
            x = np.array(x)
            predictions = esn_make_predictions(models, x, horizon)

            return np.array(predictions)

        else:

            predictions = models.predict(np.array(x))
        
        return predictions
    
    else:

        p = hyperparameters["p"]
        d = hyperparameters["d"]
        q = hyperparameters["q"]
        
        window_length = hyperparameters["window_length"]

        order = (p, d, q)
        seasonal_order = (0, 0, 0, 0)

        cfg = (order, seasonal_order, 'c')
        
        predictions = []
        ys = []
        # seed history with training dataset
        history = []
        history.extend(input_data)

        yhat = sarima_multistep_forecast(np.array(history), cfg, window_length, horizon)
        
        # store forecast in list of predictions
        predictions.append(yhat)
        
        return predictions