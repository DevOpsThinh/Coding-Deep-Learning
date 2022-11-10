# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Multiple Linear regression (Hyperspace) with Gradient Descent
#           Plot a plane that roughly approximates a dataset with three input variables.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Import the dataset
from ml.util import load_text_dataset

x1, x2, x3, y = load_text_dataset("../../../fundamentals/datasets/pizza_forester/pizza_3_vars.txt")
# These weights came out of the training phase
w = np.array([-3.98230894, 0.37333539, 1.69202346])
# Plot the axes
sns.set(rc={"axes.facecolor": "white", "figure.facecolor": "white"})
ax = plt.figure().add_subplot(projection="3d")
ax.set_zticklabels(())
ax.set_xlabel("Temperature", labelpad=15, fontsize=30)
ax.set_ylabel("Reservations", labelpad=15, fontsize=30)
ax.set_zlabel("Pizzas", labelpad=5, fontsize=30)
# Plot the data points
ax.scatter(x1, x2, y, color="b")
# Plot the plane
MARGIN = 10
edges_x = [np.min(x1) - MARGIN, np.max(x1) + MARGIN]
edges_y = [np.min(x2) - MARGIN, np.max(x2) + MARGIN]
xs, ys = np.meshgrid(edges_x, edges_y)
zs = np.array([w[0] + x * w[1] + y * w[2] for x, y in
               zip(np.ravel(xs), np.ravel(ys))])
ax.plot_surface(xs, ys, zs.reshape((2,2)), alpha=0.2)
# Display plot
plt.show()
