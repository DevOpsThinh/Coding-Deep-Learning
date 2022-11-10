# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning with linear regression - Gradient descent algorithm
#           Plot the loss on a dataset with a single input variable
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ml.util import loss, load_text_dataset

# Set the plotting DPI settings to be a bit higher.
plt.rcParams['figure.figsize'] = [8.0, 5.0]
plt.rcParams['figure.dpi'] = 150
# Import the dataset
X, Y = load_text_dataset("../../../fundamentals/datasets/pizza_forester/pizza.txt")
# Compute losses for w ranging from -1 to 4
weights = np.linspace(-1.0, 4.0, 200)
losses = [loss(X, Y, w, 0) for w in weights]
# Activate Seaborn
sns.set()
# Plot weights & losses
plt.axis([-1, 4, 0, 1000])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("weight", fontsize=30)
plt.ylabel("loss", fontsize="30")
plt.plot(weights, losses, color="black")
# Put a green cross on the minimum loss
min_index = np.argmin(losses)
plt.plot(weights[min_index], losses[min_index], "gX", markersize=26)

plt.show()
