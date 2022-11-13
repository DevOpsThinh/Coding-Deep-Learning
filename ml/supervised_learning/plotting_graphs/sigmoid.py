# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Classifications in Action.
#           Plot the sigmoid function in 2 dimensions.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ml.util import sigmoid

MARGIN_LEFT = -5
MARGIN_RIGHT = 5
# Active seaborn
sns.set()
# Configure axes
plt.axis([MARGIN_LEFT, MARGIN_RIGHT, -0.5, 1.5])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# Plot margins
plt.plot([MARGIN_LEFT, MARGIN_RIGHT], [1, 1], color="grey", linestyle="dashed")
plt.plot([MARGIN_LEFT, MARGIN_RIGHT], [0, 0], color="grey", linestyle="dashed")
# Plot sigmoid
X = np.linspace(MARGIN_LEFT, MARGIN_RIGHT, 200)
Y = [sigmoid(x) for x in X]
plt.xlabel("z", fontsize=30)
plt.ylabel("sigmoid(z)", fontsize=30)
plt.plot(X, Y, color="blue")

plt.show()



