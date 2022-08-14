# Author: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Use case: Discounted cash flow calculation

import numpy as np
from matplotlib import pyplot as plt


def calc_discounted_cash_flow(
        disc_rate, growth_rate, current_fcf, num_years, start_year):
    """
    Calculates & returns an array containing cash flow & discounted
    cash flow for each future year
    :param disc_rate: The discount rate
    :param growth_rate: The growth rate
    :param current_fcf: The current free cash flow
    :param num_years: The number of years
    :param start_year: The start year
    :return: An array containing cash flow & discounted cash flow
    """
    future_cash_flows = np.zeros(num_years)
    future_disc_cash_flows = np.zeros(num_years)

    for i in range(0, num_years):
        future_cash_flows[i] = current_fcf * (growth_rate + 1) ** i
        future_disc_cash_flows[i] = future_cash_flows[i] / ((1 + disc_rate) ** i)

    future_years = np.array(range(start_year, start_year + num_years))

    return [future_cash_flows, future_disc_cash_flows, future_years]


# Here is the cash flow data for the company (values in millions)
cash_flow_years = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
free_cash_flow = [12.05, 13.45, 10.2, 15.44, 14.9, 16.37, 17.10]
free_cash_flow_data = np.array([cash_flow_years, free_cash_flow])

growth = np.zeros(6)
for i in range(1, len(growth) + 1):
    growth[i - 1] = \
        (free_cash_flow_data[1, i] / free_cash_flow_data[1, i - 1] - 1) * 100

avg_growth_rate = growth.mean()
print('The average growth rate was: ', avg_growth_rate, '%')

[future_cash_flows, future_disc_cash_flows, future_years] = \
    calc_discounted_cash_flow(0.25, 0.08, 17, 10, 2023)
print('The future cash flow years calculated are:\n', future_years)
print('The future cash flows are:\n', future_cash_flows)
print('The discounted future cash flows are:\n', future_disc_cash_flows)

plt.plot(free_cash_flow_data[0, :], free_cash_flow_data[1, :], '-s',
         label='Past free cash flows')
plt.plot(future_years, future_cash_flows, '--d',
         label='Projected future cash flows')
plt.plot(future_years, future_disc_cash_flows, '-s',
         label='Discounted value of future cash flows')
plt.legend()
plt.grid()
plt.show()

plt.plot(free_cash_flow_data[0, :], free_cash_flow_data[1, :], '-s',
         label='Past free cash flows')
plt.plot(future_years, future_cash_flows, '--d',
         label='Projected future cash flows')
plt.plot(future_years, future_disc_cash_flows, '-s',
         label='Discounted value of future cash flows')
plt.fill_between(future_years, future_disc_cash_flows, 0, alpha=0.3)
plt.ylim([0, 35])
plt.legend()
plt.grid()
plt.show()

print('The net future value of discounted projected cash flows '
      '(the area under curve = the company value right now) is:\n',
      round(future_disc_cash_flows.sum(), 2), 'million dollars')

"""
Brief of what was done

1. Take a stock and take a look at a numpy array of previous cash flows;
2. With these cash flows as an array, compute an array with the growth values for each year with a loop;
3. From this array of growth values, compute the mean, which we will use for future cash flows;
4. Define a function that returns an array containing cash flow and discounted cash flow for each future year;
5. As each value of the discounted future cash flows is computed differently, use a loop to iterate over our array;
6. Plot the past cash flows, future cash flows, and discounted future cash flows on a graph;
7. Print out the net present value of future cash flows.
"""
