# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Let's Do Development with The Development Cycle
#          We pushed the network as far as we managed, both
#          by standardizing its input data and tuning its hyperparameters.
#          => Achieving 99% accuracy

from ml.supervised_learning.neural_networks.speeding_up_batching import the_neural_network_quieter as tnq
import mnist_standardized as standardized

tnq.train(standardized.X_train, standardized.Y_train,
          standardized.X_test, standardized.Y_test,
          _n_hidden_nodes=1200, epochs=100, batch_size=600, lr=0.8)
