# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Classifications in Action.
#           Our own MNIST library:  An MNIST loader.

import gzip
import struct

import matplotlib.pyplot as plt
import numpy as np

from ml.util import prepend_bias, one_hot_encoding

TRAIN_IMAGE = "../../../fundamentals/datasets/mnist/train-images-idx3-ubyte.gz"
TRAIN_LABEL = "../../../fundamentals/datasets/mnist/train-labels-idx1-ubyte.gz"
TEST_IMAGE = "../../../fundamentals/datasets/mnist/t10k-images-idx3-ubyte.gz"
TEST_LABEL = "../../../fundamentals/datasets/mnist/t10k-labels-idx1-ubyte.gz"


def load_images(filename):
    """Decodes images from MNIST's library files"""
    with gzip.open(filename, "rb") as f:
        _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        return all_pixels.reshape(n_images, columns * rows)


# 60000 images, each 785 (1 bias + 28 * 28) elements
X_train_data = prepend_bias(load_images(TRAIN_IMAGE))
print(X_train_data.shape)
# 10000 images, each 785 (1 bias + 28 * 28) elements
X_test_data = prepend_bias(load_images(TEST_IMAGE))
print(X_test_data.shape)

# Neural networks #
X_train = load_images(TRAIN_IMAGE)  # no bias - 784 elements
X_test = load_images(TEST_IMAGE)  # no bias - 784 elements


def load_labels(filename):
    """Loads MNIST labels into a Numpy array, then molds that array into a
        one-column matrix.
    """
    with gzip.open(filename, "rb") as f:
        f.read(8)
        all_labels = f.read()
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)


def encode_fives(y):
    """Convert all 5s to 1, and everything else to 0"""
    return (y == 5).astype(int)


def show_a_digit(num):
    """
    Use a matplotlib function to display a specific MNIST digit
    :param num: A digit will be display
    :return: None
    """
    x = load_images(TRAIN_IMAGE)
    y = load_labels(TRAIN_LABEL).flatten()
    digits = x[y == num]
    np.random.shuffle(digits)

    rows = 3
    columns = 24
    fig = plt.figure()
    for i in range(rows * columns):
        ax = fig.add_subplot(rows, columns, i + 1)
        ax.axis('off')
        ax.set_title(num, fontsize="20")
        ax.imshow(digits[i].reshape((28, 28)), cmap="gray")
    plt.show()


# 60000 labels, each with value 1 if digit is a five, and 0 otherwise
Y_train_data = encode_fives(load_labels(TRAIN_LABEL))
print(Y_train_data.shape)

Y_train_unencoded = load_labels(TRAIN_LABEL)
# 60000 labels, each consisting of 10 one-hot encoded elements
Y_train = one_hot_encoding(Y_train_unencoded, 10)

# 10000 labels
Y_test_data = encode_fives(load_labels(TEST_LABEL))
print(Y_test_data.shape)
# 10000 labels, each a single digit from 0 to 9
Y_test = load_labels(TEST_LABEL)
