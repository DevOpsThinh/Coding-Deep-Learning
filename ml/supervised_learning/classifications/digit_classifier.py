# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Classifications in Action.
#           Our own MNIST library:  A binary classifier that recognizes one of the digits in MNIST.
#                                                Or that recognizes any digit in MNIST.

import our_own_mnist_lib as data
from base_classifier import base_train
from ml.util import train, test

# The five number recognizing in MNIST
# DIGIT = 5
# w = train(data.X_train_data, data.Y_train_data, iterations=100, lr=1e-5)
# test(data.X_test_data, data.Y_test_data, w)
# data.show_a_digit(DIGIT)
# Any digit recognizing in MNIST
w = base_train(data.X_train_data, data.Y_train,
               data.X_test_data, data.Y_test,
               iterations=200, lr=1e-5)
