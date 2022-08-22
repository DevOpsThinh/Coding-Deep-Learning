# Author: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
# Loading Data Sets

import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fundamentals.custom_functions import show_img

# Load training images and labels
train_data = idx2numpy.convert_from_file('../datasets/mnist/train-images-idx3-ubyte')
train_data = np.reshape(train_data, (60000, 28 * 28))
train_label = idx2numpy.convert_from_file('../datasets/mnist/train-labels-idx1-ubyte')
print(train_data.shape)
print(train_label.shape)
# Load testing images and labels
test_data = idx2numpy.convert_from_file('../datasets/mnist/t10k-images-idx3-ubyte')
test_data = np.reshape(test_data, (10000, 28 * 28))
test_label = idx2numpy.convert_from_file('../datasets/mnist/t10k-labels-idx1-ubyte')
print(test_data.shape)
print(test_label.shape)

# use a matplotlib function to display some MNIST images
fig = plt.figure()

for i in range(9):
    img = train_data[i, :]
    img = img.reshape(28, -1)
    ax = fig.add_subplot(3, 3, i + 1)
    ax.title.set_text(str(train_label[i]))
    plt.imshow(img, cmap='gray')

plt.show(block=True)

show_img(4, '../datasets/mnist/train-images-idx3-ubyte', '../datasets/mnist/train-labels-idx1-ubyte')

# Loading the Boston House data set
raw_data = pd.read_csv('../datasets/boston.csv', header=None)
data_rows = np.reshape(raw_data.to_numpy(), (506, 14))
data = data_rows[:, :13]
target = data_rows[:, 13]
print(data.shape)
print(target.shape)
# Loading the Ames Housing data set
train_dataframe = pd.read_csv('../datasets/ames_housing/train.csv')
test_dataframe = pd.read_csv('../datasets/ames_housing/test.csv')
print(train_dataframe.shape)
print(test_dataframe.shape)
# Loading the MLF Gaussian data set
train_dataframe = pd.read_csv('../datasets/mlf_gaussian/train-gaussian.csv')
test_dataframe = pd.read_csv('../datasets/mlf_gaussian/test-gaussian.csv')
train_data = train_dataframe.to_numpy()
test_data = test_dataframe.to_numpy()
print(train_data.shape)
print(test_data.shape)
