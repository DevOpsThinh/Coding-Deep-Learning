# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Classifications in Action.
#           The binary classification: a classifier work with categorical labels.
import numpy as np

from ml.util import load_text_dataset, gradient, lower_loss, test


def train(x, _y, iterations, lr):
    _w = np.zeros((x.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, lower_loss(x, _y, _w)))
        _w -= gradient(x, _y, _w) * lr
    return _w


# Prepare data: import the dataset & classify
x1, x2, x3, y = load_text_dataset("../../../fundamentals/datasets/pizza_forester/police.txt")
X = np.column_stack((np.ones(x1.size), x1, x2, x3))
Y = y.reshape(-1, 1)
w = train(X, Y, iterations=10000, lr=0.001)
# Test it
test(X, Y, w)
