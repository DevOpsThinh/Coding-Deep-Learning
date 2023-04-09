# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning
# An MNIST data (pre-shuffled) loader that splits data into training, validation & test sets.

import numpy as np
from ml.util import load_images, load_labels, one_hot_encoding

TRAIN_IMAGE = "../../../../fundamentals/datasets/mnist/train-images-idx3-ubyte.gz"
TRAIN_LABEL = "../../../../fundamentals/datasets/mnist/train-labels-idx1-ubyte.gz"
TEST_IMAGE = "../../../../fundamentals/datasets/mnist/t10k-images-idx3-ubyte.gz"
TEST_LABEL = "../../../../fundamentals/datasets/mnist/t10k-labels-idx1-ubyte.gz"

# X_train/ X_validation/ X_test: 60k/ 5k/ 5k images
# Each image has 784 elements (28 * 28 pixels)
X_train = load_images(TRAIN_IMAGE)
X_test_all = load_images(TEST_IMAGE)  # To ensure best practice: np.random.shuffle(X_test_all)
X_validation, X_test = np.split(X_test_all, 2)

# 60K labels, each a single digit from 0 to 9
Y_train_unencoded = load_labels(TRAIN_LABEL)
#  Y_train: 60k labels, each consisting of 10 one-hot-encoded elements
Y_train = one_hot_encoding(Y_train_unencoded, 10)
# Y_validation/ Y_test: 5k/ 5k labels, each a single digit from 0 to 9
Y_test_all = load_labels(TEST_LABEL)  # To ensure best practice: np.random.shuffle(Y_test_all)
Y_validation, Y_test = np.split(Y_test_all, 2)