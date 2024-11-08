#S&P 500 - Linear Regression Prediction

import pandas as pd
import matplotlib.pyplot as plt

from statistical_models import LinRegModel

in_folder = "/STAT_4188_Final_Project"

stock = pd.read_csv(in_folder + "/sp500_10_years_data.csv", index_col="Date", parse_dates = True)

stock = stock.rename(columns={'Close/Last':'Close'})

stock = stock.iloc[::-1]

print(stock.tail(5))

print(stock.columns)

result = LinRegModel.reg_model_training(stock)

LinRegModel.linear_regression_results(result['Actual'], result['Predicted']) # model evaluation

#drawing a scatterplot if needed

# plt.scatter(range(df_pred.shape[0]),df_pred[['Predicted']],c="blue")
# plt.show()

# spotted five outliers to be dropped. - NOTE: uncomment and apply the code below if you spot severe outliers

# drop_list = df_pred.index[df_pred['Predicted'] == min(df_pred['Predicted'])].tolist()

# print("Values to be dropped at these indeces: ", drop_list)

# df_pred=df_pred.drop(index=drop_list, axis='index')

# plt.scatter(range(df_pred.shape[0]),df_pred[['Predicted']],c="blue")
# plt.show()

# creating a plot

plt.figure(figsize=(14,7))
plt.title("Regression")
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(stock['Close']) # .tail(int(len(stock['Close'])*0.5)) - add if needed
plt.plot(result['Predicted']) 
plt.plot(result['Actual']) 
plt.legend(['Predicted', 'Actual'])
plt.show()
