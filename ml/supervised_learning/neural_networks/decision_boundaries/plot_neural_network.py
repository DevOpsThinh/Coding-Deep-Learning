# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Classifications in Action.
#           Neural Networks: Are leaps & bounds more powerful than perceptrons.
#           Neural Network's Decision Boundary

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from ml.supervised_learning.neural_networks import training_the_network as tn
from ml.util import one_hot_encoding, mesh, load_text_dataset

np.random.seed(123)  # Make this code deterministic


def plot_boundary(points, _w1, _w2):
    print("Calculating boundary...")
    # Generate a grid of points over the data
    x_mesh = mesh(points[:, 0])
    y_mesh = mesh(points[:, 1])
    grid_x, grid_y = np.meshgrid(x_mesh, y_mesh)
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    # Classify points in the grid
    classifications = tn.classify(grid, _w1, _w2).reshape(grid_x.shape)
    # Trace the decision boundary
    BLUE_AND_GREEN = ListedColormap(['#BBBBFF', '#BBFFBB'])
    plt.contourf(grid_x, grid_y, classifications, cmap=BLUE_AND_GREEN)


def plot_data_by_label(input_variables, labels, label_selector, symbol):
    points = input_variables[(labels == label_selector).flatten()]
    plt.plot(points[:, 0], points[:, 1], symbol, markersize=4)


# Decide which dataset to load
x1, x2, y = load_text_dataset("../../../../fundamentals/datasets/brain_friendly/non_linearly_separable.txt")
# x1, x2, y = load_text_dataset("../../../../fundamentals/datasets/brain_friendly/circles.txt")

X_train = X_test = np.column_stack((x1, x2))
Y_train_unencoded = Y_test = y.astype(int).reshape(-1, 1)
Y_train = one_hot_encoding(Y_train_unencoded, 2)
w1, w2 = tn.train(X_train, Y_train, X_test, Y_test,
                  _n_hidden_nodes=10, iterations=100000, lr=0.3)

plot_boundary(X_train, w1, w2)
plot_data_by_label(X_train, Y_train_unencoded, 0, 'bs')
plot_data_by_label(X_train, Y_train_unencoded, 1, 'g^')
plt.gca().axes.set_xlabel("Input A", fontsize=20)
plt.gca().axes.set_ylabel("Input B", fontsize=20)
plt.gca().axes.xaxis.set_ticklabels([])
plt.gca().axes.yaxis.set_ticklabels([])
plt.show()
