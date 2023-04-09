# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Let's Do Development with The Development Cycle
#          We pushed the network as far as we managed, only by standardizing its input data.

from ml.supervised_learning.neural_networks.speeding_up_batching import the_neural_network as tn
import mnist_standardized as standardized

tn.train(standardized.X_train, standardized.Y_train, standardized.X_test, standardized.Y_test,
         _n_hidden_nodes=100, epochs=10, batch_size=256, lr=1)
