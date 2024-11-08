import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


in_folder = "/STAT_4188_Final_Project"

stock = pd.read_csv(in_folder + "/AAPL_2_years_stock_data.csv", index_col="date", parse_dates = True) # or use Date for index in case of sp500

# stock = stock.rename(columns={'Close/Last':'Close'}) #uncomment it in case of sp500

# print(stock.tail(5))

# print(stock.columns)

series = stock[['close']].copy() # or use Close

# print(series.head(5))

# Visualizing the closing prices of the data.
# plt.figure(figsize=(16,8))
# plt.title('Apple')
# plt.xlabel('Days')
# plt.ylabel('Closing Price USD ($)')
# plt.plot(series['close'])
# plt.show()

series_2 = series['close']

series_2 = pd.DataFrame(series_2)

future_days = int(len(series_2)*0.25)

series_2['prediction'] = series_2['close'].shift(-future_days)

x = np.array(series_2.drop(['prediction'], 1))[:-future_days]

y = np.array(series_2['prediction'])[:-future_days]

# Building a Decision Tree

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

tree = DecisionTreeRegressor().fit(x_train,y_train)

x_future = series_2.drop(['prediction'], 1)[:-future_days]

x_future = x_future.tail(future_days)

x_future = np.array(x_future)

# print(len(x_future))

tree_prediction = tree.predict(x_future)

# print(tree_prediction)

predictions = tree_prediction
valid = series_2[x.shape[0]:]
# print(len(valid))
# print(len(predictions))
valid['Predictions'] = predictions

# print(valid.head())

# print(valid['close'].tolist())
# print(valid['Predictions'].tolist())

# evaluate the model with MSE because it's a regression not a classification

from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(valid['close'],predictions)
mae = mean_absolute_error(valid['close'], predictions)

print("Mean Square Error(MSE): ", mse)
print("Mean Absolute Error(MSE): ", mae)

# visualize the model

plt.figure(figsize=(14,7))
plt.title("Decision Tree")
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(series_2['close'])
plt.plot(valid[['close', 'Predictions']])
plt.legend(["Original", "Valid", 'Predicted'])
plt.show()