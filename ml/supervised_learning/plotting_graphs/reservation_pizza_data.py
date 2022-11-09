# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning
# Plot the reservations/ pizzas dataset

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()
plt.axis([0, 50, 0, 50])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations", fontsize=30)
plt.ylabel("Pizzas", fontsize=30)
X, Y = np.loadtxt("../../../fundamentals/datasets/pizza.txt", skiprows=1, unpack=True)
plt.plot(X, Y, "bo")
plt.show()
