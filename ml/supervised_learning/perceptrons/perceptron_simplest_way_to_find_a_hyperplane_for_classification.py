# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Classifications in Action.
#           A primitive kind of support vector machine - Perceptron

import matplotlib.pyplot as plt
import numpy as np

# Can try two different arrangements of x
# x = np.array([[1,2], [3,6], [4,7], [5,6], [1,3], [2.5,4], [2,3]])
x = np.array([[5, 2], [3, 6], [2, 7], [1, 6], [5, 3], [3.5, 4], [4, 3]])
y_orig = np.array([-1, 1, 1, 1, -1, -1, -1])
plt.scatter(x[:, 0], x[:, 1], c=y_orig)
plt.show()

# Length of y(input) vector
L = y_orig.size
# Separate X1, X2 to see algorithm clearly
X1 = x[:, 0].reshape(L, 1)
X2 = x[:, 1].reshape(L, 1)
y = y_orig.reshape(L, 1)
# Creating our weights, start at 0 for 1st iteration
w0, w1, w2 = 1, 0, 0
# Counter to go through each point
count = 0
iteration = 1
# Learning rate
alpha = 0.01

while iteration < 1000:
    y_hat = w0 + X1 * w1 + X2 * w2
    prod = y_hat * y  # If > 1, the prediction is correct
    for p in prod:
        if p <= 1:
            # Nudge w vector in right direction
            w0 = w0 + alpha * y[count]
            w1 = w1 + alpha * y[count] * X1[count]
            w2 = w2 + alpha * y[count] * X2[count]

        count += 1
    count = 0
    iteration += 1

print('w0', w0)
print('w1', w1)
print('w2', w2)

# Final perceptron answers (If < 0 -> category 1, else > 0 -> category 2)
y = w0 + X1 * w1 + X2 * w2
# Plot the predictions via Perceptron algorithm
plt.scatter(x[:, 0], x[:, 1], c=(y < 0).reshape(1, -1)[0])
# Plot the line
q = np.array([0, 7])  # 2 points on x axis.
# Calculated hyperplane (a line)
x_ = -(w1 / w2).reshape(1, -1) * q - w0 / w2
# f(x) = w.x+b+1 support vector line
x_p = -(w1 / w2).reshape(1, -1) * q - w0 / w2 - 1 / w2
# f(x) = w.x+b+1 support vector line
x_n = -(w1 / w2).reshape(1, -1) * q - w0 / w2 + 1 / w2

plt.plot(q, x_[0])
plt.plot(q, x_p[0], 'r--')
plt.plot(q, x_n[0], 'r--')
plt.xlim([0, 6])
plt.ylim([0, 12])
plt.grid()
plt.xlabel(r'$X_{1}$')
plt.ylabel(r'$X_{2}$')
plt.show()
