# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Deep Learning: The Echidna dataset.
#           Utility functions to plot the decision boundary of a Keras model
#           over a bi-dimensional dataset.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


def plot(history):
    """
    A utility function that plots the training loss and validation loss from a Keras history object.
    :param history: A Keras history object.
    :return: None
    """
    sns.set()
    plt.plot(history.history['loss'], label='Training set', color='blue', linestyle='-')
    plt.plot(history.history["val_loss"], label='Validation set', color='green', linestyle='--')
    plt.xlabel("Epochs", fontsize=30)
    plt.ylabel("Loss", fontsize=30)
    plt.xlim(0, len(history.history['loss']))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=30)
    plt.show()


def plot_boundary(model, points):
    # Generate a grid of points over the data
    RANGE = 0.55
    x_mesh = np.arange(-RANGE, RANGE, 0.001)
    y_mesh = np.arange(-RANGE, RANGE, 0.001)
    grid_x, grid_y = np.meshgrid(x_mesh, y_mesh)
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    # Classify points in the grid
    classifications = model.predict(grid).argmax(axis=1)
    classifications_grid = classifications.reshape(grid_x.shape)
    # Trace the decision boundary
    BLUE_AND_GREEN = ListedColormap(['#BBBBFF', '#BBFFBB'])
    plt.contourf(grid_x, grid_y, classifications_grid, cmap=BLUE_AND_GREEN)


def plot_data_by_label(input_variables, labels, label_selector, symbol):
    points = input_variables[(labels == label_selector).flatten()]
    plt.plot(points[:, 0], points[:, 1], symbol, markersize=4)


def show(model, x, y, title="Decision boundary"):
    plot_boundary(model, x)
    plot_data_by_label(x, y, 0, 'bs')
    plot_data_by_label(x, y, 1, 'g^')
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.title(title, fontsize=15)
    plt.show()
