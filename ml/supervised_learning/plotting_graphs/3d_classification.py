# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Classifications in Action.
#           Plot a plane that roughly approximates a dataset with two input variables.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm

from ml.util import load_text_dataset, forward, train

# Train classifier
x1, x2, _, y = load_text_dataset("../../../fundamentals/datasets/pizza_forester/police.txt")
X = np.column_stack((np.ones(x1.size), x1, x2))
Y = y.reshape(-1, 1)
w = train(X, Y, iterations=10000, lr=0.001)
# Prepare the axes
sns.set(rc={"axes.facecolor": "white", "figure.facecolor": "white"})
ax = plt.figure().add_subplot(projection="3d")
ax.set_xlabel("Reservations", labelpad=15, fontsize=30)
ax.set_ylabel("Temperature", labelpad=15, fontsize=30)
ax.set_zlabel("Police Call", labelpad=15, fontsize=30)
# Configure the data points
ax.scatter(x1, x2, y, color="r")
# Plot the model
MARGIN = 3
MESH_SIZE = 20
x, y = np.meshgrid(np.linspace(x1.min() - MARGIN, x1.max() + MARGIN, MESH_SIZE),
                   np.linspace(x2.min() - MARGIN, x2.max() + MARGIN, MESH_SIZE))
z = np.array([forward(np.column_stack(([1], [i], [j])), w) for i, j in
              zip(np.ravel(x), np.ravel(y))])
z = z.reshape((MESH_SIZE, MESH_SIZE))
ax.plot_surface(x, y, z, alpha=0.75, cmap=cm.winter,
                linewidth=0, antialiased=True, color="black")

plt.show()
