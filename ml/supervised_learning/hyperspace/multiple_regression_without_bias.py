# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning with linear regression - Hyperspace
import numpy as np

from ml.supervised_learning.hyperspace.common import train
from ml.util import load_text_dataset

# Import the dataset & train model
x1, x2, x3, y = load_text_dataset("../../../fundamentals/datasets/pizza_forester/pizza_3_vars.txt")
X = np.column_stack((x1, x2, x3))
Y = y.reshape(-1, 1)
w = train(X, Y, iterations=100000, lr=0.001)
