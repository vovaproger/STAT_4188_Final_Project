#Linear Regression for Stok Prediction

import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import metrics

in_folder = "/STAT_4188_Final_Project"

stock = pd.read_csv(in_folder + "/AAPL_2_years_stock_data.csv") # OHLC (Open-High-Low-Close) format

print(stock.head(5))

print(stock.columns)

# creating a time series plot for our closing prices

# stock['close'].plot(figsize=(10, 7))
# plt.title("AAPL stock closing prices", fontsize = 17)
# plt.ylabel('Price', fontsize=14)
# plt.xlabel('Time', fontsize=14)
# plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
# plt.show()

# training and testing our linear regression model

def model_train_test_sets (stock_data, col='close'):

    assert col in stock_data.columns, "The indicated column is not in the dataset"

    y_pred_set = stock_data[col] # we select our Y-variable as our predicted column
    x_set = stock_data.drop(col, axis=1) # we select the rest of our data after dropping the chosen Y-variable column
    
    data_len = stock_data.shape[0]

    print("Total Length of All Histories: ",  str(data_len))

    train_split = int(data_len * 0.85) # we need a lot of data to train our model

    print("Training length: ", train_split)

    val_split = train_split + int(data_len * 0.1)

    print("Validation length: ", str(int(data_len * 0.1)))

    print("Test length: ", str(data_len-val_split))

    X_train, X_val, X_test = x_set[:train_split], x_set[train_split:val_split], x_set[val_split:]
    Y_train, Y_val, Y_test = y_pred_set[:train_split], y_pred_set[train_split:val_split], y_pred_set[val_split:]

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

X_train, X_val, X_test, Y_train, Y_val, Y_test = model_train_test_sets(stock.drop('date', axis=1)) # don't forget to drop date column, as it is of a string class

l_reg = LinearRegression()
l_reg.fit(X_train, Y_train)

print("Model Performance: ", str(l_reg.score(X_train, Y_train)))

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

#making predictions

Y_train_pred = l_reg.predict(X_train)
Y_val_pred = l_reg.predict(X_val)
Y_test_pred = l_reg.predict(X_test)

#analyzing predictions

linear_regression_results(Y_train,Y_train_pred)
linear_regression_results(Y_val,Y_val_pred)
linear_regression_results(Y_test,Y_test_pred)

#creating an intermediate dataframe reflecting the actual and the predicted values

left = Y_train_pred.shape[0]
right = Y_train_pred.shape[0] + Y_val_pred.shape[0]
dates = list(stock['date'])

assert len(dates[left:right]) == len(list(Y_val_pred)) and len(dates[left:right]) == len(Y_val.values), "lists of dates, predicted, and actual values are of different lengths"

pred_actual_stock_data = {
    "Date": list(dates[left:right]),
    "Predicted": list(Y_val_pred),
    "Actual": Y_val.values
}

df_pred = pd.DataFrame(pred_actual_stock_data)

print(df_pred.head(5)) # see how it looks

# creating a plot

df_pred[['Predicted','Actual']].plot()
plt.show()