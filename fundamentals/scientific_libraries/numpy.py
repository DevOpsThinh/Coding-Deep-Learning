# Author: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Package & libraries for scientific computing section
from matplotlib import pyplot as plt

import numpy as np

# Mathematical functions
print(np.log(32))
print(np.exp(3))
print(np.sin(2))
print(np.cos(2))

# Numpy arrays: Single dimensional
my_arr = np.array([0, 1, 3, 4, 5])
print(my_arr)

my_arr = np.array(range(0, 10))
print(my_arr)

my_arr = np.linspace(1.1, 4.8, num=6)
print(my_arr)
print("\nmy array type is: ", type(my_arr))
print("\nThe first element is: ", my_arr[0])

my_arr[0] = 1.0
print("\nThe first element is now: ", my_arr[0])

my_arr[0:4] = 99
print('\nThe my array is now:', my_arr)
print('\nThe max of my array is: ', my_arr.max())
print('\nThe min of my array is: ', my_arr.min())
print('The mean of my array is: ', my_arr.mean())
# Numpy arrays: Multi-dimensional
_2d_matrix = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
print(_2d_matrix)
print('\nThe matrix dimensions are: ', _2d_matrix.shape)
print('\nElement in first row, third column is: ', _2d_matrix[0, 2])
print('\nElement in second row, fifth column is: ', _2d_matrix[1, 4])

rs_2d_matrix = _2d_matrix.reshape(5, 2)
print('\nReshaped matrix is: \n', rs_2d_matrix)
trans_2d_matrix = _2d_matrix.transpose()
print('\nTransposed matrix is: \n', trans_2d_matrix)
into_lines = _2d_matrix.ravel()
print('\nUnraveled matrix is: \n', into_lines)

array_zeros = np.zeros((3, 5))
print('\n', array_zeros)
array_ones = np.ones((8, 9))
print('\n', array_ones)
array_pi = np.full((3, 7), 3.14159)
print('\n', array_pi)

_3d_matrix = np.ones((3, 5, 3))
print('\n 3D matrix: \n', _3d_matrix)

array_ones[2:6, 2:] = 8.95
print('My large matrix of ones is now: \n', array_ones)

arr_1 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr_2 = np.array([[3, 6, 9], [12, 9, 15], [6, 36, 63]])
print('Array 1:\n', arr_1)
print('Array 2: \n', arr_2)
array_3 = arr_2 + np.sin(arr_1) - arr_1 / arr_2
print('\nArray 3:\n', array_3)

other_array = np.array([1, 2, 3, 4, 5])
for i in other_array:
    print(i**3 + i)

x = np.array(range(0, 9))
y = np.sqrt(x)
plt.plot(x, y)
print('\nMy an other array y has values: ', y)

