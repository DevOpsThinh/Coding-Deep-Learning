# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning with linear regression
import numpy as np

from ml.util import load_text_dataset


def predict(x, w):
    """
    Predicts the pizzas from the reservations
    :param x: The input variable
    :param w: The weight
    :return: The label (the pizzas)
    """
    return x * w


def loss(x, y, w):
    """
    Calculate the mean squared error
    :param x: The input variable
    :param y: The label (the pizzas)
    :param w: The weight
    :return: The mean squared error
    """
    return np.average((predict(x, w) - y) ** 2)


def train(x, y, iterations, lr):
    w = 0
    for i in range(iterations):
        c_loss = loss(x, y, w)
        print("Iteration %4d => Loss: %.6f" % (i, c_loss))

        if loss(x, y, w + lr) < c_loss:
            w += lr
        elif loss(x, y, w - lr) < c_loss:
            w -= lr
        else:
            return w
    raise Exception("Couldn't converge within %d iterations" % iterations)


# Import the dataset
X, Y = load_text_dataset("../../../fundamentals/datasets/pizza_forester/pizza.txt")
# Training phase: train the system
W = train(X, Y, iterations=10000, lr=0.01)
print("\nw=%.3f" % W)
# Predict phase: predict the number of pizzas
print("Prediction: x=%d => y=%.2f" % (20, predict(20, W)))
