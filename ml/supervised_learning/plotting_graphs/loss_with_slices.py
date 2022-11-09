# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning with linear regression - Gradient descent algorithm
#           Plot the loss on a dataset with two input variables, and the slices of its
#           partial derivatives.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm

from ml.util import loss, load_text_dataset, train_with_history

# Import the dataset, train model
X, Y = load_text_dataset("../../../fundamentals/datasets/pizza.txt")
w, b, _history = train_with_history(X, Y, iterations=100000, lr=0.001,
                                    precision=0.000001, init_w=-0, init_b=0)
# Prepare history
history = np.array(_history)
history_w = history[:, 0]
history_b = history[:, 1]
history_loss = history[:, 2]
# Prepare matrices for 3D plot
MESH_SIZE = 20
weights = np.linspace(np.min(history_w) - 10, np.max(history_w) + 10, MESH_SIZE)
biases = np.linspace(np.min(history_b) - 100, np.max(history_b) + 100, MESH_SIZE)
W, B = np.meshgrid(weights, biases)
losses = np.array([loss(X, Y, w, b) for w, b in zip(np.ravel(W), np.ravel(B))])
L = losses.reshape(MESH_SIZE, MESH_SIZE)
# Plot surface
sns.set(rc={"axes.facecolor": "white", "figure.facecolor": "white"})
ax = plt.figure().add_subplot(projection="3d")
ax.set_zticklabels(())
ax.set_xlabel("Weight", labelpad=20, fontsize=30)
ax.set_ylabel("Bias", labelpad=20, fontsize=30)
ax.set_zlabel("Loss", labelpad=5, fontsize=30)
ax.plot_surface(W, B, L, cmap=cm.Blues, linewidth=0, antialiased=True)
# Trace the partial derivative "slices"
plt.plot(weights, [history_b[0] for w in weights],
         [loss(X, Y, w, history_b[0]) for w in weights], color="r")
plt.plot([history_w[0] for b in biases], biases,
         [loss(X, Y, history_w[0], b) for b in biases], color="r")
# Display plot
plt.show()
