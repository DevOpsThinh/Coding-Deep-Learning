# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Classifications in Action.
#           Neural Networks: Are leaps & bounds more powerful than perceptrons.
#           The Network's classification: A Classifier's Answers
#           A neural network implementation

import numpy as np

from ml.supervised_learning.neural_networks import training_the_network as tn


def prepare_batches(x_train, y_train, batch_size):
    """Splitting our pre-shuffled MNIST dataset into batches"""
    x_batches = []
    y_batches = []
    n_examples = x_train.shape[0]

    for batch in range(0, n_examples, batch_size):
        batch_end = batch + batch_size
        x_batches.append(x_train[batch:batch_end])
        y_batches.append(y_train[batch:batch_end])

    return x_batches, y_batches


def report(epoch, batch, x_train, y_train, x_test, y_test, _w1, _w2):
    """To check how well the system is learning"""
    y_hat, _ = tn.forward(x_train, _w1, _w2)
    training_loss = tn.loss(y_train, y_hat)
    classifications = tn.classify(x_test, _w1, _w2)
    accuracy = np.average(classifications == y_test) * 100.0
    print("%5d-%d > Loss: %.8f, Accuracy: %.2f%%" % (epoch, batch, training_loss, accuracy))


def train(x_train, y_train, x_test, y_test, _n_hidden_nodes, epochs, batch_size, lr):
    n_input_variables = x_train.shape[1]
    n_classes = y_train.shape[1]
    # Initialize all the weights at zero
    # _w1 = np.zeros((n_input_variables + 1, _n_hidden_nodes))
    # _w2 = np.zeros((_n_hidden_nodes + 1, n_classes))
    # Initialize all the weights with good initialization
    _w1, _w2 = tn.initialize_weights(n_input_variables, _n_hidden_nodes, n_classes)
    x_batches, y_batches = prepare_batches(x_train, y_train, batch_size)

    for e in range(epochs):
        for i in range(len(x_batches)):
            y_hat, h = tn.forward(x_batches[i], _w1, _w2)
            w1_gradient, w2_gradient = tn.back(x_batches[i], y_batches[i], y_hat, _w2, h)
            _w1 = _w1 - (w1_gradient * lr)
            _w2 = _w2 - (w2_gradient * lr)

            report(e, i, x_train, y_train, x_test, y_test, _w1, _w2)
    return _w1, _w2


if __name__ == "__main__":
    from ml.supervised_learning.classifications import our_own_mnist_lib as mnist
    w1, w2 = train(mnist.X_train, mnist.Y_train, mnist.X_test, mnist.Y_test, _n_hidden_nodes=200, epochs=2, batch_size=20000, lr=0.01)