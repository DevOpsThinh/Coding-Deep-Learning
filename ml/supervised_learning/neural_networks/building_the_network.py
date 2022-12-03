# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Classifications in Action.
#           Neural Networks: Are leaps & bounds more powerful than perceptrons.
#           The Network's classification: A Classifier's Answers

import json

import numpy as np

from ml.supervised_learning.classifications import our_own_mnist_lib as mnist
from ml.util import sigmoid, prepend_bias


def loss(y, y_hat):
    """The log loss - cross entropy loss for our binary classifier"""
    return -np.sum(y * np.log(y_hat)) / y.shape[0]


def softmax(logits):
    """The Softmax function"""
    exponentials = np.exp(logits)
    return exponentials / np.sum(exponentials, axis=1).reshape(-1, 1)


def forward(x, w1, w2):
    """
    Forward propagation.
    Calculates the system's outputs from the system's inputs.
    """
    # Calculate the hidden layer
    h = sigmoid(np.matmul(prepend_bias(x), w1))
    # Calculate the output layer
    y_hat = softmax(np.matmul(prepend_bias(h), w2))
    return y_hat


def classify(x, w1, w2):
    """Predict the value of an unlabeled image"""
    y_hat = forward(x, w1, w2)
    labels = np.argmax(y_hat, axis=1)
    return labels.reshape(-1, 1)


def report(iteration, x_train, y_train, x_test, y_test, w1, w2):
    """To check how well the system is learning"""
    y_hat = forward(x_train, w1, w2)
    training_loss = loss(y_train, y_hat)
    classifications = classify(x_test, w1, w2)
    accuracy = np.average(classifications == y_test) * 100.0
    print("Iteration: %5d, Loss: %.6f, Accuracy: %.2f%%" % (iteration, training_loss, accuracy))


# Time travel testing section #
with open('weights.json') as f:
    weights = json.load(f)

weight1, weight2 = (np.array(weights[0]), np.array(weights[1]))
report(0, mnist.X_train, mnist.Y_train, mnist.X_test, mnist.Y_test, weight1, weight2)
# Iteration: 0, Loss: 2.221439, Accuracy: 43.19%
