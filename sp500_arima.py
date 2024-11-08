#S&P 500 - ARIMA

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from statistical_models import ARIMA_Model

in_folder = "/STAT_4188_Final_Project"

stock = pd.read_csv(in_folder + "/sp500_10_years_data.csv", index_col="Date", parse_dates = True)

stock = stock.rename(columns={'Close/Last':'Close'})

stock = stock.iloc[::-1] # reverses the initial dataset

print(stock.tail(5))

print(stock.columns)

series = stock[['Close']].copy()

train_rate = int(len(series['Close']) * 0.75)

test_set_index = series['Close'][train_rate:].index

arima = ARIMA_Model(series,'Close',train_rate)

d_and_series = arima.arima_d_param_selection()

series_diff = d_and_series[1]

d = d_and_series[0]

print("the p-value is ", arima.test_stationarity(series_diff))

print("the differencing value is ", d)

series_diff = series_diff.dropna()

ar_ma_lags = arima.arima_ar_ma_selection(series_diff)

p = ar_ma_lags[0]
q = ar_ma_lags[1]

print("AR lags p: ", p)
print("MA lags q: ", q)

training_result = arima.arima_training(p,d,q)

training_dict = {"actual": training_result[0], "predicted": training_result[1]}

arima.arima_evaluation(training_dict["predicted"], training_dict["actual"])

# visualization 

plt.figure(figsize=(12,6))
plt.plot(series.index[-int(train_rate):], series['Close'].tail(int(train_rate)), color='green', label = 'Train Stock Price')
plt.plot(test_set_index, training_dict["actual"], color='red', label='Real Stock Price')
plt.plot(test_set_index, training_dict["predicted"], color='blue', label='Predicted Stock Price')
plt.xticks(rotation=30)
plt.legend()
plt.show()