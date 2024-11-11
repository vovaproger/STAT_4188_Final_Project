# Stock data 

import requests
import pandas as pd

# API key and parameters
API_KEY = 'YOUR_API_KEY'
ticker = 'S&P500'  
multiplier = 1
timespan = 'day'  # Choose from 'minute', 'hour', 'day', 'week', 'month', 'year'
start_date = 2021
end_date = 2024

#fetching data using Polygon API
def fetch_data(from_date, to_date, ticker):
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}?apiKey={API_KEY}'
    response = requests.get(url)
    return response.json().get('results', [])

# Display the data, if needed
# print(data)

#extracting results for long ranges

all_stock_data = []

for year in range(start_date, end_date):
    from_date = f'{year}-01-01'
    to_date = f'{year}-12-31'

    stock_data = fetch_data(from_date, to_date, ticker)
    all_stock_data.extend(stock_data)

#converting stamps for readable data

out_folder = "/STAT_4188_Final_Project/"

if all_stock_data:
    for record in all_stock_data:
        record['date'] = pd.to_datetime(record['t'], unit='ms').strftime('%Y-%m-%d')  # Convert 't' (timestamp) to a readable date

    df = pd.DataFrame(all_stock_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    df.to_csv(out_folder + f'{ticker}_{end_date-start_date-1}_years_stock_data.csv', index=False)

    print(f'Data saved to {out_folder}{ticker}_{end_date-start_date-1}_years_stock_data.csv')
else:
    print("No data found for the given parameters.")
