# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Classifications in Action.
#           Plot a datasets' losses calculated with the MSE and log loss formulae.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm

from ml.util import forward, sigmoid

# Come up with some plausible data
n_examples = 20
np.random.seed(1234)
X = np.random.rand(n_examples, 2)
w = [1.23, 2.34]
Y = np.dot(X,w)
Y_binary = np.greater_equal(Y, np.median(Y)).astype(int)
# Prepare plot
MESH_SIZE = 400
sns.set(rc={"axes.facecolor": "white", "figure.facecolor": "white"})


def prepare_axes():
    ax = plt.figure().add_subplot(projection="3d")
    ax.clear()
    ax.set_zticklabels(())
    ax.set_xlabel("w1", labelpad=15, fontsize=30)
    ax.set_ylabel("w2", labelpad=15, fontsize=30)
    ax.set_zlabel("Loss", labelpad=15, fontsize=30)
    ax.view_init(30, 120)
    return ax


def log_loss(x, y, _w):
    y_hat = sigmoid(np.matmul(x, _w))
    f_term = y * np.log(y_hat)
    s_term = (1 - y) * np.log(1 - y_hat)
    return -np.average(f_term + s_term)


def mse_loss(x, y, _w):
    return np.average((forward(x, _w) - y) ** 2)


# Plot mean squared error loss
axes = prepare_axes()
plt.title("Mean Squared Error Loss", fontsize=30)
view_range = 100  # Wide range to see the creeks & valleys
W1, W2 = np.meshgrid(np.linspace(-view_range, view_range, MESH_SIZE),
                     np.linspace(-view_range, view_range, MESH_SIZE))
losses = np.array([mse_loss(X, Y_binary, [w1, w2]) for w1, w2 in
                   zip(np.ravel(W1), np.ravel(W2))])
L = losses.reshape((MESH_SIZE, MESH_SIZE))
axes.plot_surface(W1, W2, L, cmap=cm.autumn, edgecolors="black",
                  vmax=np.max(L), linewidth=0.2, antialiased=True)
plt.ion()
plt.show()
input("Press<Enter>...")

# Plot logarithmic loss
axes = prepare_axes()
plt.title("Log Loss", fontsize=30)
view_range = 20  # Tight range to avoid overflows
W1, W2 = np.meshgrid(np.linspace(-view_range, view_range, MESH_SIZE),
                     np.linspace(-view_range, view_range, MESH_SIZE))
losses = np.array([log_loss(X, Y_binary, [w1, w2]) for w1, w2 in
                   zip(np.ravel(W1), np.ravel(W2))])
L = losses.reshape((MESH_SIZE, MESH_SIZE))
axes.plot_surface(W1, W2, L, cmap=cm.autumn, edgecolors="black",
                  vmax=np.max(L), linewidth=0.2, antialiased=True)
plt.show()
input("Enter to exit...")
