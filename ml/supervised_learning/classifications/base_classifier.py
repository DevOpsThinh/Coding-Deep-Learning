# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Classifications in Action.
#           The Classifier's Answers

import numpy as np
from ml.util import forward, gradient


def classify(x, w):
    y_hat = forward(x, w)
    labels = np.argmax(y_hat, axis=1)
    return labels.reshape(-1, 1)


def loss(x, y, w):
    y_hat = forward(x, w)
    first_term = y * np.log(y_hat)
    second_term = (1 - y) * np.log(1 - y_hat)
    return -np.sum(first_term + second_term) / x.shape[0]


def report(iteration, x_train, y_train, x_test, y_test, w):
    matches = np.count_nonzero(classify(x_test, w) == y_test)
    n_test_examples = y_test.shape[0]
    matches = matches * 100.0 / n_test_examples
    training_loss = loss(x_train, y_train, w)
    print("%d - Loss: %.20f, %.2f%%" % (iteration, training_loss, matches))


def base_train(x_train, y_train, x_test, y_test, iterations, lr):
    w = np.zeros((x_train.shape[1], y_train.shape[1]))
    for i in range(iterations):
        report(i, x_train, y_train, x_test, y_test, w)
        w -= gradient(x_train, y_train, w) * lr
    report(iterations, x_train, y_train, x_test, y_test, w)
    return w
