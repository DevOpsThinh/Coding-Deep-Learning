# Author: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
from math import sqrt


# Custom functions
def get_data_from_list(stocks):
    """
    Get a list of:
    First 3 items in list, first and last item, number of items
    :param stocks: A stocks list
    :return: A new stocks list
    """
    size = len(stocks)
    another_stocks = stocks

    print('The first item is: ', another_stocks[0])
    print('The last item is: ', another_stocks[size - 1])
    print('The number of items is: ', size)

    return [another_stocks[:3], another_stocks[0], another_stocks[size - 1], size]


def ret_first_n_stocks(stocks, n):
    """
    Get elements of a stocks list
    :param stocks: A stocks list
    :param n: The element's index
    :return: All items up to index n
    """
    return stocks[:n]


def tuple_distance(tuple1, tuple2):
    """
    Calculate the Cartesian distance between two given points.
    d = sqrt ((x1 - x2)^2 +(y1 - y2)^2)
    :param tuple1: point (x1, y1)
    :param tuple2: point (x2, y2)
    :return: The Cartesian distance
    """
    return sqrt((tuple1[0] - tuple2[0]) ** 2 + (tuple1[1] - tuple2[1]) ** 2)


def print_basic_arithmetic(result = 'I would like to have a result of arithmetic expression please.'):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Result: {result}')  # Press Ctrl+F8 to toggle the breakpoint.
