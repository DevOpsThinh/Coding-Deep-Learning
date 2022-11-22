# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Package & libraries for scientific computing section
# Scikit-learn - a library: Simple and efficient tools for predictive data analysis
# Ref: https://scikit-learn.org/stable/
#
# Topic: Supervised Learning: Classifications in Action.
#           Scikit-learn Linear support vector classifier - works better than our Perceptron

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC

# Set the plotting DPI settings to be a bit higher.
plt.rcParams['figure.figsize'] = [8.0, 5.0]
plt.rcParams['figure.dpi'] = 150

# Linear support vector classifier - works better than our Perceptron
x, y = make_blobs(n_features=2, centers=2, n_samples=100, random_state=3)
plt.scatter(x[:, 0], x[:, 1], marker='o', c=y, s=25, edgecolors='k')
plt.show()

svm_classifier = LinearSVC(C=1, tol=1e-8, max_iter=1e5).fit(x, y)
# Prediction
predictions = svm_classifier.predict(x)
plt.scatter(x[:, 0], x[:, 1], marker='o', c=predictions, s=25, edgecolors='k')

w = svm_classifier.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-7, 3)
yy = a * xx - (svm_classifier.intercept_[0]) / w[1]
plt.plot(xx, yy)
plt.show()

# Linear support vector classifier - with overlapping points
x_, y_ = make_blobs(n_features=2, centers=2, n_samples=100, random_state=4)
plt.scatter(x_[:, 0], x_[:, 1], marker='o', c=y_, s=25, edgecolors='k')
svm_classifier_ = LinearSVC().fit(x_, y_)
# Prediction
predictions_ = svm_classifier_.predict(x_)
plt.scatter(x_[:, 0], x_[:, 1], marker='o', c=y_, s=25, edgecolors='k')  # c=predictions

w_ = svm_classifier_.coef_[0]
a_ = -w_[0] / w_[1]
xx_ = np.linspace(5, 15)
yy_ = a_ * xx_ - (svm_classifier_.intercept_[0]) / w_[1]
plt.plot(xx_, yy_)
plt.show()