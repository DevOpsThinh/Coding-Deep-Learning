# Author: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Use case: viewing & analysis of the financials of corporate America
# Ref: https://pandas-datareader.readthedocs.io/en/latest/

import pandas as pd
import pandas_datareader.data as pdr
from fundamentals.custom_functions import make_the_graph

start = pd.to_datetime('2020-12-01')
end = pd.to_datetime('2021-12-01')
ticker_data = pdr.DataReader('SPY', 'yahoo', start, end)

make_the_graph(ticker_data['Open'], 'Yahoo finance', 'Date', 'SPY value')

ticker_data
# Select specific rows (use the index in a few ways)
ticker_data.iloc[-5:]
ticker_data.iloc[5:]
ticker_data.iloc[200]
ticker_data.loc['2021-06-01']

make_the_graph(ticker_data.loc[ticker_data.index > '2021-06-01']['Close'], 'Yahoo finance', 'Date', 'SPY value')
