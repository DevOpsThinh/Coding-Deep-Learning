# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: The zen of Testing
#           A neural network implementation

# An MNIST data (pre-shuffled) loader that splits data into training, validation & test sets.
import matplotlib.pyplot as plt
import numpy as np
from ml.supervised_learning.neural_networks import training_the_network as tn
from ml.util import load_images, load_labels, one_hot_encoding

TRAIN_IMAGE = "../../../fundamentals/datasets/mnist/train-images-idx3-ubyte.gz"
TRAIN_LABEL = "../../../fundamentals/datasets/mnist/train-labels-idx1-ubyte.gz"
TEST_IMAGE = "../../../fundamentals/datasets/mnist/t10k-images-idx3-ubyte.gz"
TEST_LABEL = "../../../fundamentals/datasets/mnist/t10k-labels-idx1-ubyte.gz"

# X_train/ X_validation/ X_test: 60k/ 5k/ 5k images
# Each image has 784 elements (28 * 28 pixels)
X_train = load_images(TRAIN_IMAGE)
X_test_all = load_images(TEST_IMAGE)  # To ensure best practice: np.random.shuffle(X_test_all)
X_validation, X_test = np.split(X_test_all, 2)

# 60K labels, each a single digit from 0 to 9
Y_train_unencoded = load_labels(TRAIN_LABEL)
#  Y_train: 60k labels, each consisting of 10 one-hot-encoded elements
Y_train = one_hot_encoding(Y_train_unencoded, 10)
# Y_validation/ Y_test: 5k/ 5k labels, each a single digit from 0 to 9
Y_test_all = load_labels(TEST_LABEL)  # To ensure best practice: np.random.shuffle(Y_test_all)
Y_validation, Y_test = np.split(Y_test_all, 2)


# This loss() takes different parameters than the ones in other source files
def loss(_x, _y, _w1, _w2):
    _y_hat, _ = tn.forward(_x, _w1, _w2)
    return -np.sum(_y * np.log(_y_hat)) / _y.shape[0]


def train(x_train, y_train, x_test, y_test, _n_hidden_nodes, iterations, lr):
    n_input_variables = x_train.shape[1]
    n_classes = y_train.shape[1]
    # Initialize all the weights at zero
    # _w1 = np.zeros((n_input_variables + 1, _n_hidden_nodes))
    # _w2 = np.zeros((_n_hidden_nodes + 1, n_classes))
    # Initialize all the weights with good initialization
    _w1, _w2 = tn.initialize_weights(n_input_variables, _n_hidden_nodes, n_classes)
    _training_losses = []
    _test_losses = []

    for i in range(iterations):
        y_hat_train, h = tn.forward(x_train, _w1, _w2)
        y_hat_test, _ = tn.forward(x_test, _w1, _w2)
        w1_gradient, w2_gradient = tn.back(x_train, y_train, y_hat_train, _w2, h)
        _w1 = _w1 - (w1_gradient * lr)
        _w2 = _w2 - (w2_gradient * lr)

        training_loss = -np.sum(y_train * np.log(y_hat_train)) / y_train.shape[0]
        _training_losses.append(training_loss)
        test_loss = -np.sum(y_test * np.log(y_hat_test)) / y_test.shape[0]
        _test_losses.append(test_loss)

        print("%5d > Training loss: %.5f - Test loss: %.5f" % (i, training_loss, test_loss))

    return _training_losses, _test_losses, _w1, _w2


training_losses, test_losses, w1, w2 = train(X_train, Y_train,
                                             X_test,
                                             one_hot_encoding(Y_test, 10),
                                             _n_hidden_nodes=200,
                                             iterations=10000,
                                             lr=0.01)
training_accuracy = tn.accuracy(X_train, Y_train, w1, w2)
test_accuracy = tn.accuracy(X_test, Y_test, w1, w2)
print("Training accuracy: %.2f%%, Test accuracy: %.2f%%" % (training_accuracy, test_accuracy))

plt.plot(training_losses, label="Training set", color='blue', linestyle='-')
plt.plot(test_losses, label="Test set", color='green', linestyle='--')
plt.xlabel("Iterations", fontsize=30)
plt.ylabel("Loss", fontsize=30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=30)
plt.show()

# Results:
# 9996 > Training loss: 0.14478 - Test loss: 0.15406
# 9997 > Training loss: 0.14478 - Test loss: 0.15406
# 9998 > Training loss: 0.14477 - Test loss: 0.15405
# 9999 > Training loss: 0.14476 - Test loss: 0.15405
# Training accuracy: 10.06%, Test accuracy: 95.46%
