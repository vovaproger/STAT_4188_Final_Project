#S&P 500 - Decision Trees Prediction

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statistical_models import DecisionTreesModelAugmented

in_folder = "/STAT_4188_Final_Project"

stock = pd.read_csv(in_folder + "/sp500_10_years_data.csv", index_col="Date", parse_dates = True)

stock = stock.rename(columns={'Close/Last':'Close'})

stock = stock.iloc[::-1] # reverses the initial dataset 

price = stock[['Close']].copy()

price_2 = price['Close']

price_2 = pd.DataFrame(price_2) # working dataset

future_days = int(len(price_2)*0.25)

price_2['Prediction'] = price_2['Close'].shift(-future_days)

x = np.array(price_2.drop(columns='Prediction'))[:-future_days]

y = np.array(price_2['Prediction'])[:-future_days]

trees = DecisionTreesModelAugmented(future_days, np.round(future_days/len(price_2), 1))

valid_data = trees.trees_training(x,y,price_2)

trees.trees_evaluation(valid_data['Predictions'], valid_data['Close'])

# visualize the model

plt.figure(figsize=(14,7))
plt.title("Decision Tree")
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(price_2['Close'])
plt.plot(valid_data[['Close', 'Predictions']])
plt.legend(["Original", "Valid", 'Predicted'])
plt.show()

