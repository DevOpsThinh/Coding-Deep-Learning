# Author: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Package & libraries for scientific computing section
# Pandas: To read & write data to your operating system & to analyse & manipulate data.
# Ref: https://pandas.pydata.org/pandasdocs/stable/reference/index.html
# Ref: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
import pandas as pd
from matplotlib import pyplot as plt

student_dict = {
    'StuCode': ['19211tt3065', '19211tt3030', '19211tt3020'],
    'Name': ['Thinh', 'Thang', 'Thien'],
    'Age': [29, 22, 21]
}

student_df = pd.DataFrame(student_dict)
student_df.to_csv('it_student_k19.csv')
my_data = pd.read_csv('it_student_k19.csv', index_col=0)

income_data = pd.read_csv('us-income-annual.csv', delimiter=';')
print('DataFrame shape is: ', income_data.shape)
print(income_data.keys())
print('\nRevenue:\n', income_data['Revenue'])  # view a single column series
print('\nThe resulting type of object:\n', type(income_data['Report Date']))
print('\nUsing .values command after our series selection turns this into a:\n',
      type(income_data['Report Date'].values))

income_data_set = set(income_data['Ticker'].values)
print(income_data_set)
print(income_data['Ticker'].unique())

is_it_2020 = (income_data['Fiscal Year'] == 2020)
print('The boolean logic on the Fiscal Year column heading is: ', is_it_2020)
income_data[is_it_2020]

income_data.sort_values(by=['Net Income'])
income_data['Net Income'].hist()
# Plotting the data for only fiscal year 2020
income_data[income_data['Fiscal Year'] == 2020]['Net Income'].hist(bins=100, log=True)
plt.title('USA Corporate Net Income for 2020 Histogram')
plt.xlabel('Value in USD currency')
plt.ylabel('Number of Instances')
plt.grid()
plt.show()
print('Max value is: ', income_data[income_data['Fiscal Year'] == 2020]['Net Income'].max())
income_data.describe()
income_data_2020 = income_data[income_data['Fiscal Year'] == 2020]
