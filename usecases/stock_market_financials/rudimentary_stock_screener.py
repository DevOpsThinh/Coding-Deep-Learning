# Author: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Use case: Rudimentary stock filtering for price/ earnings, price sales ratios, etc.

import matplotlib.pyplot as plt
import pandas as pd

# Get csv data
income_data = pd.read_csv(
    'D:/PythonInPractice/Coding-Deep-Learning/fundamentals/scientific_libraries/'
    'us-income-annual.csv', delimiter=';')
stock_prices = pd.read_csv('us-shareprices-daily.csv', delimiter=';')
# Retrieve info
print('Income data size is: ', income_data.shape)
print('Stock prices data size is: ', stock_prices.shape)
print('Income keys are: ', income_data.keys())
print('Stock prices keys are: ', stock_prices.keys())
print(income_data.head(7))
print(stock_prices.head(7))

income_data['Publish Date'] = pd.to_datetime(income_data['Publish Date'])

# Plot a histogram
income_data[income_data['Fiscal Year'] == 2020]['Publish Date'].hist(bins=100)
plt.title('USA Corporate Net Income for 2020 Histogram')
plt.show()

income_data_2021 = income_data[income_data['Fiscal Year'] == 2021]

(income_data[income_data['Fiscal Year'] == 2020]['Ticker']).unique()

stock_prices_2021 = (stock_prices[stock_prices['Date'] == '2021-03-01'])
print('stock_prices_2021 dataframe shape is: ', stock_prices_2021)
print(stock_prices_2021.head())
# To first cut down the share prices dataframe
stock_prices_2021 = \
    stock_prices_2021[stock_prices_2021['Ticker'].isin(income_data_2021['Ticker'])]
stock_prices_2021

income_data_2021 = \
    income_data_2021[income_data_2021['Ticker'].isin(stock_prices_2021['Ticker'])]
income_data_2021

stock_prices_2021 = stock_prices_2021.sort_values(by=['Ticker'])
income_data_2021 = income_data_2021.sort_values(by=['Ticker'])

stock_data_2021 = income_data_2021
stock_data_2021['Market Cap'] = \
    income_data_2021['Shares (Diluted)'].values * stock_prices_2021['Open'].values

stock_data_2021['Market Cap'].hist(bins=100)
plt.show()

stock_data_2021['Price/ Earnings'] = \
    stock_data_2021['Market Cap'] / stock_data_2021['Net Income']
stock_data_2021['Price/ Earnings'].hist(bins=100, log=True)
plt.show()

stock_data_2021[((stock_data_2021['Price/ Earnings'] >= 5) &
                 (stock_data_2021['Price/ Earnings'] <= 50))]['Price/ Earnings'].hist()
plt.show()

stock_data_2021[((stock_data_2021['Price/ Earnings'] >= 5) &
                 (stock_data_2021['Price/ Earnings'] <= 50))]['Market Cap'].hist(log=True)
plt.show()

print(stock_data_2021.keys())
# Make scatter plot of Net Income vs. Market Cap
stock_data_2021[((stock_data_2021['Price/ Earnings'] >= 5) &
                 (stock_data_2021['Price/ Earnings'] <= 50))]\
    .plot.scatter(x='Market Cap', y='Net Income')
plt.show()

"""
Brief of what was done

1. Lets get the stock data;
2. Make a Market Cap series;
3. Using market cap, make a price/ earnings series in the stock_data DataFrame;
4. Filter your stock data;
5. Plot & view the histogram;
6. Plot scatter plots of Net Income vs. Market Cap.
"""