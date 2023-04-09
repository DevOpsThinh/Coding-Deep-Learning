# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Data Standardization
#           Compare the network's accuracy on MNIST with and without standardization.

from ml.supervised_learning.neural_networks.development_cycle import regular_mnist as regular
from ml.supervised_learning.neural_networks.speeding_up_batching import the_neural_network as tn
import mnist_standardized as standardized

print("Regular MNIST: ")
tn.train(regular.X_train, regular.Y_train, regular.X_validation,
         regular.Y_validation, _n_hidden_nodes=200, epochs=2, batch_size=60, lr=0.1)
# 0-999 > Loss: 0.46027237, Accuracy: 82.12%
# 1-999 > Loss: 0.50322436, Accuracy: 80.12%

print("Standardized MNIST: ")
tn.train(standardized.X_train, standardized.Y_train, standardized.X_validation,
         standardized.Y_validation, _n_hidden_nodes=200, epochs=2, batch_size=60, lr=0.1)
# 0-999 > Loss: 0.27886302, Accuracy: 89.38%
# 1-999 > Loss: 0.21286223, Accuracy: 91.72%
