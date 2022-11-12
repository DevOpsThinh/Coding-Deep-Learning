# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Classifications in Action.
#           The binary classification: a classifier work with categorical labels.
import numpy as np

from ml.util import load_text_dataset


def sigmoid(z):
    """The logistic function: z = 1 / (1 + e^-z)"""
    return 1 / (1 + np.exp(-z))


def forward(x, _w):
    wei_sum = np.matmul(x, _w)
    return sigmoid(wei_sum)


def classify(x, _w):
    return np.round(forward(x, _w))


def loss(x, _y, _w):
    y_hat = forward(x, _w)
    first_term = _y * np.log(y_hat)
    second_term = (1 - _y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)


def gradient(x, _y, _w):
    return np.matmul(x.T, (forward(x, _w) - _y)) / x.shape[0]


def train(x, _y, iterations, lr):
    _w = np.zeros((x.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, loss(x, _y, _w)))
        _w -= gradient(x, _y, _w) * lr
    return _w


def test(x, _y, _w):
    total_examples = x.shape[0]
    correct_results = np.sum(classify(x, _w) == _y)
    success_percent = correct_results * 100 / total_examples
    print("\nSuccess: %d/%d (%.2f%%)" % (correct_results, total_examples, success_percent))


# Prepare data: import the dataset & classify
x1, x2, x3, y = load_text_dataset("../../../fundamentals/datasets/pizza_forester/police.txt")
X = np.column_stack((np.ones(x1.size), x1, x2, x3))
Y = y.reshape(-1, 1)
w = train(X, Y, iterations=10000, lr=0.001)
# Test it
test(X, Y, w)
