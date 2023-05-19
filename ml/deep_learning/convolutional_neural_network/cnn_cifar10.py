# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Deep Learning: Convolutional Neural Networks (CNNs)
#           A CNN that trains on Keras's CIFAR-10 images dataset

import numpy as np

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical

from ml.deep_learning.neural_network_keras.decision_boundaries import decision_boundary_2dimensional as boundary

# Hyperparameters we can adjust
DROPOUT_PROBABILITY = 0.5

(X_train_raw, Y_train_raw), (X_test_raw, Y_test_raw) = cifar10.load_data()

X_train = X_train_raw / 255
X_test_all = X_test_raw / 255
X_validation, X_test = np.split(X_test_all, 2)
Y_train_encoded = to_categorical(Y_train_raw)
Y_validation_encoded, Y_test = np.split(to_categorical(Y_test_raw), 2)

# Create a sequential model
model = Sequential()

# Convolutional layers
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(BatchNormalization())  # To improve accuracy
model.add(Dropout(DROPOUT_PROBABILITY))  # To reduce over-fitting

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(DROPOUT_PROBABILITY))

# Four-dimensional tensor => bi-dimensional matrix of flat data.
model.add(Flatten())

# Fully connected layers - Dense
model.add(Dense(1000, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(DROPOUT_PROBABILITY))

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(DROPOUT_PROBABILITY))

model.add(Dense(10, activation='softmax'))

#  Compile a model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)

# Train a network
history = model.fit(
    X_train,
    Y_train_encoded,
    validation_data=(X_validation, Y_validation_encoded),
    epochs=20,
    batch_size=32
)

# Draw a decision boundary
boundary.plot(history)
