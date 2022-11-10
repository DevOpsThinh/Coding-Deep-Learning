# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning with linear regression - Gradient descent algorithm
import numpy as np

from ml.util import loss, load_text_dataset, predict


def gradient(x, y, wei):
    """ Calculate the gradient of the curve """
    return 2 * np.average(x * (predict(x, wei, 0) - y))


def train(x, y, iterations, lr):
    wei = 0
    for i in range(iterations):
        print("Iteration %4d => Loss: %.10f" % (i, loss(x, y, wei, 0)))
        wei -= gradient(x, y, wei) * lr
    return wei


X, Y = load_text_dataset("../../fundamentals/datasets/pizza_forester/pizza.txt")
# Training phase: train the system
w = train(X, Y, iterations=100, lr=0.001)
print("\nw=%.10f" % w)
