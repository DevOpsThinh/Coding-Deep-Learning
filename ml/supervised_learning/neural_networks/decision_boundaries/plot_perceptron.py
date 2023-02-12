# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Classifications in Action.
#           Neural Networks: Are leaps & bounds more powerful than perceptrons.
#           Perceptron's Decision Boundary
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from ml.supervised_learning.classifications import base_classifier
from ml.util import one_hot_encoding, mesh, prepend_bias

from ml.util import load_text_dataset


def plot_boundary(points, _w):
    """
    The plot_boundary() functionality were inspired by the
    documentation of the BSD-licensed scikit-learn library.
    """
    print("Calculating boundary...")
    # Generate a grid of points over the data
    x_mesh = mesh(points[:, 1])
    y_mesh = mesh(points[:, 2])
    grid_x, grid_y = np.meshgrid(x_mesh, y_mesh)
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    # Classify points in the grid
    classifications = base_classifier.classify(
        prepend_bias(grid), _w).reshape(grid_x.shape)
    # Trace the decision boundary
    BLUE_AND_GREEN = ListedColormap(['#BBBBFF', '#BBFFBB'])
    plt.contourf(grid_x, grid_y, classifications, cmap=BLUE_AND_GREEN)


def plot_data_by_label(input_variables, labels, label_selector, symbol):
    points = input_variables[(labels == label_selector).flatten()]
    plt.plot(points[:, 1], points[:, 2], symbol, markersize=4)


# Decide which dataset to load
x1, x2, y = load_text_dataset("../../../../fundamentals/datasets/brain_friendly/linearly_separable.txt")

X_train = X_test = prepend_bias(np.column_stack((x1, x2)))
Y_train_unencoded = Y_test = y.astype(int).reshape(-1, 1)
Y_train = one_hot_encoding(Y_train_unencoded, 2)
w = base_classifier.base_train(X_train, Y_train, X_test, Y_test, iterations=10000, lr=0.1)

plot_boundary(X_train, w)
plot_data_by_label(X_train, Y_train_unencoded, 0, 'bs')
plot_data_by_label(X_train, Y_train_unencoded, 1, 'g^')

plt.gca().axes.set_xlabel("Input A", fontsize=20)
plt.gca().axes.set_ylabel("Input B", fontsize=20)
plt.gca().axes.xaxis.set_ticklabels([])
plt.gca().axes.yaxis.set_ticklabels([])
plt.show()
