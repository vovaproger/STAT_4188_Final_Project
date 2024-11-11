import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler # scaling data

class LinRegModel:
    def reg_model_training(stock_data, col='Close', train_s_value=0.75):

        assert col in stock_data.columns, "The indicated column is not in the dataset"

        # assert (train_s_value, val_s_value) > (0,0) and (train_s_value, val_s_value) < (1,1) and train_s_value + val_s_value < 1, "splitting values are out of range"

        y_pred_set = stock_data[col] # we select our Y-variable as our predicted column
        x_set = stock_data.drop(col, axis=1) # we select the rest of our data after dropping the chosen Y-variable column

        data_len = stock_data.shape[0]

        print("Total Length of All Histories: ",  str(data_len))

        train_split = int(data_len * train_s_value)

        print("Training length: ", train_split)

        # val_split = train_split + int(data_len * val_s_value)

        # print("Validation length: ", str(int(data_len * val_s_value)))

        print("Test length: ", str(data_len))

        # X_train, X_val, X_test = x_set[:train_split], x_set[train_split:val_split], x_set[val_split:]
        # Y_train, Y_val, Y_test = y_pred_set[:train_split], y_pred_set[train_split:val_split], y_pred_set[val_split:]

        x_train, x_test, y_train, y_test = x_set[:train_split], x_set[train_split:], y_pred_set[:train_split], y_pred_set[train_split:]

        l_reg = LinearRegression().fit(x_train, y_train)

        y_pred = l_reg.predict(x_test)

        result_data = pd.DataFrame()

        result_data['Actual'] = y_test

        result_data['Predicted'] = y_pred

        return result_data

    def linear_regression_results(y_true, y_pred):
        # Regression metrics
        explained_variance=metrics.explained_variance_score(y_true, y_pred)
        mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
        # mse=metrics.mean_squared_error(y_true, y_pred) 
        # mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
        # median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
        r2=metrics.r2_score(y_true, y_pred)

        # because sklearn's MSE is "deprecated," we need to use a numerical method for calculating MSE and MSLE

        n = len(y_true) # or len(y_pred)
        sum_sq_diff = np.sum((np.array(y_true)-np.array(y_pred))**2)
        mse = sum_sq_diff / n

        sm_sq_log_diff = np.sum((np.log(y_pred+1)-np.log(y_true+1))**2)
        mean_squared_log_error = sm_sq_log_diff / n

        print('explained_variance: ', round(explained_variance,4))    
        print('mean_squared_log_error: ', round(mean_squared_log_error,4))
        print('r2: ', round(r2,4))
        print('MAE: ', round(mean_absolute_error,4))
        print('MSE: ', round(mse,4))
        print('RMSE: ', round(np.sqrt(mse),4))
        print(' ')
        
class ARIMA_Model:
    def __init__(self,series,col,train_rate):
        self.series = series
        self.col = col
        self.train_rate = train_rate

    def arima_ar_ma_selection(self, differenced_series):

        acf_values = acf(differenced_series, nlags=20)
        pacf_values = pacf(differenced_series, nlags=20)

        plot_acf(differenced_series)
        plot_pacf(differenced_series)
        plt.show()

        significant_acf_lags = [lag for lag, value in enumerate(acf_values) if abs(value) > 1.96 / (len(differenced_series)**0.5)]
        significant_pacf_lags = [lag for lag, value in enumerate(pacf_values) if abs(value) > 1.96 / (len(differenced_series)**0.5)]

        p = significant_pacf_lags[0]
        q = significant_acf_lags[0]

        return p, q
    def arima_d_param_selection(self): # figure it out later
        d = 0
        series_diff = self.series[self.col]
        p_value = self.test_stationarity(self.series[self.col].dropna())
        if  p_value < 0.05:
            return 0, series_diff
        while p_value >= 0.05:
            series_diff = series_diff.diff().dropna()
            p_value = self.test_stationarity(series_diff)
            d+=1
        return d, series_diff
    
    def test_stationarity(self, series):
        test_result = adfuller(series)
        p_value = test_result[1]
        # print('ADF-Statistic: ', test_result[0])
        # print('p-value: ', test_result[1])
        return p_value  # p-value
    
    def arima_training(self, p, d, q): 
        train_set, test_set = self.series[self.col][:self.train_rate], self.series[self.col][self.train_rate:]

        history = [x for x in train_set]

        y = list(test_set)

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

        return y, predictions
    
    def arima_evaluation(self, predicted_values, actual_values):
        n = len(actual_values)

        sum_sq_diff = np.sum((np.array(actual_values)-np.array(predicted_values))**2)

        mse = sum_sq_diff / n

        mae = mean_absolute_error(actual_values, predicted_values)

        rmse = np.sqrt(mse)

        print("MSE="+str(mse))
        print("MAE="+str(mae))
        print("RMSE="+str(rmse))

class DecisionTreesModelAugmented:
    def __init__(self, future_days, test_size = 0.2):
        self.test_size = test_size,
        self.future_days = future_days

    def trees_training(self, x, y, price):

        x_train, _, y_train, _ = train_test_split(x, y, test_size = self.test_size[0])

        tree = DecisionTreeRegressor().fit(x_train,y_train)

        x_future = price.drop(columns='Prediction')[:-self.future_days]

        x_future = x_future.tail(self.future_days)

        x_future = np.array(x_future)

        tree_prediction = tree.predict(x_future)

        valid = price[x.shape[0]:]

        valid['Predictions'] = tree_prediction

        return valid
    
    def trees_evaluation(self, y_pred, y_test):
        mse = mean_squared_error(y_test,y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print("Mean Square Error(MSE): ", mse)
        print("Mean Absolute Error(MSE): ", mae)