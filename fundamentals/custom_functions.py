# Author: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
from math import sqrt
import pickle, os


# Custom functions
def file_with_pickle_data(file_name):
    """
    Pickling data for more efficient to use a machine-readable binary file.
    :param file_name: Name of the file
    :return: None
    """
    if not os.path.isfile(file_name):
        data = [0, 1]
        data[0] = input('Enter topic: ')
        data[1] = input('Enter series: ')

        file = open(file_name, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        file = open(file_name, 'rb')
        data = pickle.load(file)
        file.close()
        print('\nWelcome back to: ', data[0], '-', data[1])


def update_a_file_with_block(file_name, mode, new_content, position):
    """
    Updating file strings
    :param file_name: Name of the file
    :param mode: File mode
    :param new_content: The new content will be updated
    :param position: The current file position
    :return: None
    """
    with open(file_name, mode) as file:
        text = file.read()
        print('\nString: ', text)
        print('\nPosition in file now: ', file.tell())
        file.seek(position)
        file.write(new_content)
        file.seek(0)
        text = file.read()
        print('\nString: ', text)
        print('\nPosition in file now: ', file.tell())


def write_a_file_with_block(file_name, mode, content):
    """
    Writing files: Adds content to the file & display the file's current status
    in the "with" block.
    :param file_name: Name of the file
    :param mode: File mode
    :param content: The content will be written
    :return: None
    """
    with open(file_name, mode) as file:
        file.write(content)
        print('\nFile Now Closed?: ', file.closed)
    print('File Now Closed?: ', file.closed)


def re_write_a_file(file_name):
    """
    Re-write a file to append a citation
    :param file_name: Name of the file
    :return: None
    """
    file = open(file_name, 'a')
    file.write('(by Truong Thinh)')
    file.close()


def read_a_file(file_name, mode):
    """
    Reading files: Read the entire contents
    :param file_name: Name of the file
    :param mode: File mode
    :return: None
    """
    file = open(file_name, mode)
    for line in file:
        # Do something here.
        print(line, end='')
    file.close()


def write_a_file(file_name, mode, content):
    """
    Writing files: Adds content to the file
    :param file_name: Name of the file
    :param mode: File mode
    :param content: The content will be written
    :return: None
    """
    file = open(file_name, mode)
    file.write(content)
    file.close()


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
