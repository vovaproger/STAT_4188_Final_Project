#AAPL - ARIMA Prediction

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

in_folder = "/STAT_4188_Final_Project"

stock = pd.read_csv(in_folder + "/AAPL_2_years_stock_data.csv", index_col="date", parse_dates = True) # or use Date for index in case of sp500

# stock = stock.rename(columns={'Close/Last':'Close'}) #uncomment it in case of sp500

# print(stock.tail(5))

# print(stock.columns)

series = stock[['close']].copy() # or use Close

print(series.head(5))

# series.plot()
# plt.xticks(rotation=30)
# plt.show() # uncomment to see how the series data plot

# plotting autocorrelation

from statsmodels.tsa.stattools import adfuller # checking for stationarity

def test_stationarity(series):
    test_result = adfuller(series)
    # print('ADF-Statistic: ', test_result[0])
    # print('p-value: ', test_result[1])
    return(test_result[1]) # p-value

# test_stationarity(series['close'].dropna()) # p-value is 0.78, which is way to big, so we must apply differencing

# test_stationarity(series['close'].diff().dropna()) # p-value is way less than 0.05. Select d = 1.

def arima_d_param_selection(series,col):
    d = 0
    series_diff = series[col]
    p_value = test_stationarity(series[col].dropna())
    if  p_value < 0.05:
        return 0
    while p_value >= 0.05:
        series_diff = series_diff.diff().dropna()
        p_value = test_stationarity(series_diff)
        d+=1
    return d, series_diff

d = arima_d_param_selection(series,'close')[0]

print("the differencing value is ", arima_d_param_selection(series,'close')[0])

series_diff = arima_d_param_selection(series,'close')[1]

series_diff = series_diff.dropna()

# identifying significant AR and MA coefficients:

from statsmodels.tsa.stattools import acf, pacf

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def arima_ar_ma_selection(differenced_series):

    acf_values = acf(differenced_series, nlags=20)
    pacf_values = pacf(differenced_series, nlags=20)

    significant_acf_lags = [lag for lag, value in enumerate(acf_values) if abs(value) > 1.96 / (len(differenced_series)**0.5)]
    significant_pacf_lags = [lag for lag, value in enumerate(pacf_values) if abs(value) > 1.96 / (len(differenced_series)**0.5)]

    p = significant_pacf_lags[0]
    q = significant_acf_lags[0]

    return p, q

print("AR lags p: ", arima_ar_ma_selection(series_diff)[0])
print("MA lags q: ", arima_ar_ma_selection(series_diff)[1])

# plot_acf(series['close'])
# plot_pacf(series['close'])
# plt.show() # uncomment for lag diagnostics

# modeling ARIMA

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

p = arima_ar_ma_selection(series_diff)[0]
q = arima_ar_ma_selection(series_diff)[1]
# d = arima_d_param_selection(series,'close')[0]

train_rate = int(len(series['close']) * 0.9)

train_set, test_set = series['close'][:train_rate], series['close'][train_rate:]

history = [x for x in train_set]

y = test_set

predictions = []

arima = ARIMA(history, order=(p,d,q)).fit()

yhat = arima.forecast()[0]

predictions.append(yhat)

history.append(y[0]) # making first prediction

for i in range(1, len(y)): # rolling prediction
    arima = ARIMA(history, order=(p,d,q)).fit()

    yhat = arima.forecast()[0]

    predictions.append(yhat)

    obs = y[i]

    history.append(obs)

# model evaluation 

n = len(y)

sum_sq_diff = np.sum((np.array(y)-np.array(predictions))**2)

mse = sum_sq_diff / n

mae = mean_absolute_error(y, predictions)

rmse = np.sqrt(mse)

print("MSE="+str(mse))
print("MAE="+str(mae))
print("RMSE="+str(rmse))

# visualization 

plt.figure(figsize=(12,6))
plt.plot(series.index[-int(train_rate*0.5):], series['close'].tail(int(train_rate*0.5)), color='green', label = 'Train Stock Price')
plt.plot(test_set.index, y, color='red', label='Real Stock Price')
plt.plot(test_set.index, predictions, color='blue', label='Predicted Stock Price')
plt.xticks(rotation=30)
plt.show()