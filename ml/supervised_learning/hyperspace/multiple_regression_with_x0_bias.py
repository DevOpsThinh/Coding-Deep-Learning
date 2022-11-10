# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning with linear regression - Hyperspace
import numpy as np

from ml.supervised_learning.hyperspace.common import train, predict
from ml.util import load_text_dataset

# Preparing data
x1, x2, x3, y = load_text_dataset("../../../fundamentals/datasets/pizza_forester/pizza_3_vars.txt")
X = np.column_stack((np.ones(x1.size), x1, x2, x3))
Y = y.reshape(-1, 1)
# Training phase
w = train(X, Y, iterations=100000, lr=0.001)
print("\nWeights: %s" % w.T)    # w transposed
# Predict phase
print("\nA few predictions:")
for i in range(5):
    print("X[%d] -> %.4f (label: %d)" % (i, predict(X[i], w), Y[i]))
