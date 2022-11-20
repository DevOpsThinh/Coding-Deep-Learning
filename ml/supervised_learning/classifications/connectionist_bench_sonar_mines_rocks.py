# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Classifications in Action.
#           Our own Sonar library:  An Sonar loader.
#            Minesweeper Challenge: A binary classifier that discriminate between sonar signals
#                       bounced off a metal cylinder and those bounced off a roughly cylindrical rock.
# ML Repo: https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)

import csv

import numpy as np

from ml.util import one_hot_encoding

# Load the entire dataset into a large (208, 61) NumPy array
TRAIN_FILE = "../../../fundamentals/datasets/sonar/sonar.all-data"
DATA_LIST = list(csv.reader(open(TRAIN_FILE)))
DATA_WITHOUT_BIAS = np.array(DATA_LIST)

# Prepend a bias column, resulting in a (208, 62) matrix:
DATA = np.insert(DATA_WITHOUT_BIAS, 0, 1, axis=1)
# Shuffle data
np.random.seed(1234)
np.random.shuffle(DATA)
# Extract a (208, 61) input matrix
X = DATA[:, 0:-1].astype(np.float32)
# Extract a (208, 1) matrix of labels
labels = DATA[:, -1].reshape(-1, 1)
Y_unencoded = (labels == 'M').astype(np.int32)
# Split into training & test set
SIZE_OF_TRAINING_SET = 160
X_train, X_test = np.vsplit(X, [SIZE_OF_TRAINING_SET])
Y_train_unencoded, Y_test = np.vsplit(Y_unencoded, [SIZE_OF_TRAINING_SET])
# One hot encode the training set
Y_train = one_hot_encoding(Y_train_unencoded, 2)
