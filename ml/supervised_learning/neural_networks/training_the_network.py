# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Classifications in Action.
#           Neural Networks: Are leaps & bounds more powerful than perceptrons.
#           The Network's classification: A Classifier's Answers
#           A neural network implementation

import numpy as np

# from ml.supervised_learning.classifications import our_own_mnist_lib as mnist
from ml.util import sigmoid, prepend_bias


def train(x_train, y_train, x_test, y_test, _n_hidden_nodes, iterations, lr):
    n_input_variables = x_train.shape[1]
    n_classes = y_train.shape[1]
    # Initialize all the weights at zero
    # _w1 = np.zeros((n_input_variables + 1, _n_hidden_nodes))
    # _w2 = np.zeros((_n_hidden_nodes + 1, n_classes))
    # Initialize all the weights with good initialization
    _w1, _w2 = initialize_weights(n_input_variables, _n_hidden_nodes, n_classes)

    for i in range(iterations):
        y_hat, h = forward(x_train, _w1, _w2)
        w1_gradient, w2_gradient = back(x_train, y_train, y_hat, _w2, h)
        _w1 = _w1 - (w1_gradient * lr)
        _w2 = _w2 - (w2_gradient * lr)

        report(i, x_train, y_train, x_test, y_test, _w1, _w2)

    return _w1, _w2


def report(iteration, x_train, y_train, x_test, y_test, _w1, _w2):
    """To check how well the system is learning"""
    y_hat, _ = forward(x_train, _w1, _w2)
    training_loss = loss(y_train, y_hat)
    classifications = classify(x_test, _w1, _w2)
    _accuracy = np.average(classifications == y_test) * 100.0
    print("Iteration: %5d, Loss: %.8f, Accuracy: %.2f%%" % (iteration, training_loss, _accuracy))


def initialize_weights(_n_input_variables, _n_hidden_nodes, _n_classes):
    """
    Initializing the weights with random values.
    :param _n_input_variables:
    :param _n_hidden_nodes:
    :param _n_classes:
    :return: The value of weights
    """
    w1_rows = _n_input_variables + 1
    _w1 = np.random.randn(w1_rows, _n_hidden_nodes) * np.sqrt(1 / w1_rows)

    w2_rows = _n_hidden_nodes + 1
    _w2 = np.random.randn(w2_rows, _n_classes) * np.sqrt(1 / w2_rows)

    return _w1, _w2


def accuracy(_x, _y_unencoded, _w1, _w2):
    return np.average(classify(_x, _w1, _w2) == _y_unencoded) * 100.0


def classify(_x, _w1, _w2):
    """Predict the value of an unlabeled image"""
    y_hat, _ = forward(_x, _w1, _w2)
    labels = np.argmax(y_hat, axis=1)
    return labels.reshape(-1, 1)


def back(_x, _y, _y_hat, _w2, _h):
    """
    The Backpropagation.
    That calculates the gradients of the weights in our neural network.
    """
    w2_gradient = np.matmul(prepend_bias(_h).T, (_y_hat - _y)) / _x.shape[0]
    w1_gradient = np.matmul(prepend_bias(_x).T, np.matmul(_y_hat - _y, _w2[1:].T) * sigmoid_gradient(_h)) / _x.shape[0]
    return w1_gradient, w2_gradient


# def back(X, Y, y_hat, w2, h):
#     w2_gradient = np.matmul(prepend_bias(h).T, y_hat - Y) / X.shape[0]
#
#     a_gradient = np.matmul(y_hat - Y, w2[1:].T) * sigmoid_gradient(h)
#     w1_gradient = np.matmul(prepend_bias(X).T, a_gradient) / X.shape[0]
#
#     return (w1_gradient, w2_gradient)

def forward(_x, _w1, _w2):
    """
    Forward propagation.
    Calculates the system's outputs from the system's inputs.
    The network calculates each layer from the previous one.
    """
    h = sigmoid(np.matmul(prepend_bias(_x), _w1))
    y_hat = softmax(np.matmul(prepend_bias(h), _w2))
    return y_hat, h


def loss(_y, _y_hat):
    """The log loss - cross entropy loss for our binary classifier"""
    return -np.sum(_y * np.log(_y_hat)) / _y.shape[0]


def sigmoid_gradient(_sigmoid):
    return np.multiply(_sigmoid, (1 - _sigmoid))


def softmax(logits):
    """The Softmax function"""
    exponentials = np.exp(logits)
    return exponentials / np.sum(exponentials, axis=1).reshape(-1, 1)

# w1, w2 = train(mnist.X_train, mnist.Y_train, mnist.X_test, mnist.Y_test, _n_hidden_nodes=200, iterations=10000, lr=0.01)
