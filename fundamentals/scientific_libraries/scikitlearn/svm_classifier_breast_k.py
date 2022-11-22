# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Package & libraries for scientific computing section
# Scikit-learn - a library: Simple and efficient tools for predictive data analysis
# Ref: https://scikit-learn.org/stable/
#
# Topic: Supervised Learning: Classifications in Action.
#           Scikit-learn Linear support vector classifier - for breast cancer (k)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#  Load the breast cancer dataset
k = load_breast_cancer()
X = k.data  # All the features
Y = k.target  # All the labels
X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, test_size=0.2, random_state=20)

svm_classifier = SVC().fit(X_train, Y_train)
# Prediction
predictions = svm_classifier.predict(X_test)
# Print confusion Matrix & Classification Report
c_matrix = np.array(confusion_matrix(Y_test, predictions, labels=[1, 0]))
confusion = pd.DataFrame(c_matrix, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer', 'predicted_healthy'])
print(confusion)  # Show the performance of classification.
print(classification_report(Y_test, predictions))

# Set the plotting DPI settings to be a bit higher.
plt.rcParams['figure.figsize'] = [8.0, 5.0]
plt.rcParams['figure.dpi'] = 150

# Plot the breast cancer data analysis
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel('predicted_healthy')
plt.ylabel('predicted_cancer')
plt.title('Breast Cancer Data Analysis')
plt.show()
