# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Classifications in Action.
#           Neural Networks: Are leaps & bounds more powerful than perceptrons.
#           Neural Network's model function
#           Plot the model function of a perceptron on a dataset with two input variables

from matplotlib import cm
from ml.util import load_text_dataset, one_hot_encoding
from ml.supervised_learning.neural_networks import training_the_network as tn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()

np.random.seed(123)  # Make this code deterministic

# Decide which dataset to load
x1, x2, y = load_text_dataset("../../../../fundamentals/datasets/brain_friendly/non_linearly_separable.txt")
# x1, x2, y = load_text_dataset("../../../../fundamentals/datasets/brain_friendly/circles.txt")

# Train classifier
X_train = X_test = np.column_stack((x1, x2))
Y_train_unencoded = Y_test = y.astype(int).reshape(-1, 1)
Y_train = one_hot_encoding(Y_train_unencoded, 2)
w1, w2 = tn.train(X_train, Y_train, X_test, Y_test,
                  _n_hidden_nodes=10, iterations=100000, lr=0.3)

# Plot the axes
sns.set(rc={"axes.facecolor": "white", "figure.facecolor": "white"})
ax = plt.figure().add_subplot(projection="3d")
ax.set_zticks([0, 0.5, 1])
ax.set_xlabel("Input A", labelpad=15, fontsize=30)
ax.set_ylabel("Input B", labelpad=15, fontsize=30)
ax.set_zlabel("ŷ", labelpad=5, fontsize=30)

# Plot the data points
blue_squares = X_train[(Y_train_unencoded == 0).flatten()]
ax.scatter(blue_squares[:, 0], blue_squares[:, 1], 0, c='b', marker='s')
green_triangles = X_train[(Y_train_unencoded == 1).flatten()]
ax.scatter(green_triangles[:, 0], green_triangles[:, 1], 1, c='g', marker='^')

# Plot the model
MARGIN = 0.5
MESH_SIZE = 1000  # This model has a lot of detail, so we need a hi-res mesh

x, y = np.meshgrid(np.linspace(x1.min() - MARGIN, x1.max() + MARGIN, MESH_SIZE),
                   np.linspace(x2.min() - MARGIN, x2.max() + MARGIN, MESH_SIZE))
grid = zip(np.ravel(x), np.ravel(y))

# Calculate all the outputs of forward(), in the format (y_hat, h):
forwards = [tn.forward(np.column_stack(([i], [j])), w1, w2) for i, j in grid]

# For each (y_hat, y), keep only the second column of y_hat:
z = np.array([y_hat for y_hat, h in forwards])[:, 0, 1]
z = z.reshape((MESH_SIZE, MESH_SIZE))
ax.plot_surface(x, y, z, alpha=0.75, cmap=cm.winter, linewidth=0, antialiased=True)
plt.show()
