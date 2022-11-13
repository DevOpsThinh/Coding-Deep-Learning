# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Classifications in Action.
#           Plot an example of classification.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ml.util import load_text_dataset, train, forward

# Train classifier
x1, _, _, y = load_text_dataset("../../../fundamentals/datasets/pizza_forester/police.txt")
X = np.column_stack((np.ones(x1.size), x1))
Y = y.reshape(-1, 1)
w = train(X, Y, iterations=10000, lr=0.001)

sns.set()  # Activate seaborn

plt.plot(X, Y, "bo")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations", fontsize=30)
plt.ylabel("Police Call", fontsize=30)
MARGIN = 3
left_edge = X[:, 1].min() - MARGIN
right_edge = X[:, 1].max() + MARGIN
inputs = np.linspace(left_edge - MARGIN, right_edge + MARGIN, 2500)
x_values = np.column_stack((np.ones(inputs.size), inputs.reshape(-1, 1)))

# rounding introduced by classify()
y_values = forward(x_values, w)   # no rounding
# y_values = classify(x_values, w)  # rounded
plt.axis([left_edge - MARGIN, right_edge + MARGIN, -0.05, 1.05])
plt.plot(x_values[:, 1], y_values, color="g")
plt.show()
