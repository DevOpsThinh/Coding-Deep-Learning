# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Multiple Linear regression (Hyperspace) with Gradient Descent

import numpy as np


def predict(x, w):
    return np.matmul(x, w)


def loss(x, y, w):
    return np.average((predict(x, w) - y) ** 2)


def gradient(x, y, w):
    return 2 * np.matmul(x.T, (predict(x, w) - y)) / x.shape[0]


def train(x, y, iterations, lr):
    w = np.zeros((x.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, loss(x, y, w)))
        w -= gradient(x, y, w) * lr
    return w
