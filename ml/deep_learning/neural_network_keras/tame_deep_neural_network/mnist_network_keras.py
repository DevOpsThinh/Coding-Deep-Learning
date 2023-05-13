# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Deep Learning: Taming Deep Networks
#           MNIST classifier (10 epochs & stochastic GD) - a deep neural network written in Keras.

import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical

from ml.deep_learning.neural_network_keras.decision_boundaries import decision_boundary_2dimensional as boundary

(X_train_raw, Y_train_raw), (X_test_raw, Y_test_raw) = mnist.load_data()

X_train = X_train_raw.reshape(X_train_raw.shape[0], -1) / 255
X_test_all = X_test_raw.reshape(X_test_raw.shape[0], -1) / 255
X_validation, X_test = np.split(X_test_all, 2)
Y_train_encoded = to_categorical(Y_train_raw)
Y_validation_encoded, Y_test = np.split(to_categorical(Y_test_raw), 2)

# Create a sequential model
model = Sequential()
model.add(Dense(1200, activation='sigmoid'))
model.add(Dense(500, activation='sigmoid'))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

#  Compile a model
model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(learning_rate=0.1),
    metrics=['accuracy']
)

# Train a network
history = model.fit(
    X_train,
    Y_train_encoded,
    validation_data=(X_validation, Y_validation_encoded),
    epochs=10,
    batch_size=32
)
# Epoch 10/10 1875/1875 [==============================] - 7s 4ms/step - loss: 0.2017 - accuracy: 0.9391 - val_loss:
# 0.2472 - val_accuracy: 0.9212

# Draw a decision boundary
boundary.plot(history)
