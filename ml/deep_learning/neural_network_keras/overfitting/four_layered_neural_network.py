# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Deep Learning: The Echidna dataset.
#           Building the echidna - four-layered neural network with over-fitting

import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.utils import to_categorical

from ml.deep_learning.neural_network_keras.decision_boundaries import decision_boundary_2dimensional as boundary
from ml.util import load_seeded_shuffled_text_dataset

# Load the Echidna dataset

current_directory = os.path.dirname(__file__)
text_dataset = os.path.join(current_directory, "echidna.txt")

X, Y = load_seeded_shuffled_text_dataset(text_dataset)

X_train, X_validation, X_test = np.split(X, 3)
Y_train, Y_validation, Y_test = np.split(Y, 3)

Y_train_encoded = to_categorical(Y_train)
Y_validation_encoded = to_categorical(Y_validation)

# Create a sequential model
model = Sequential()
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

#  Compile a model
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

# Train a network
history = model.fit(X_train, Y_train_encoded, validation_data=(X_validation, Y_validation_encoded), epochs=30000,
                    batch_size=25)
# Epoch 30000/30000 12/12 [==============================] - 0s 2ms/step - loss: 0.0036 - accuracy: 1.0000 -
# val_loss: 0.7884 - val_accuracy: 0.8877 37813/37813 [==============================] - 17s 441us/step

# Draw a decision boundary
boundary.show(model, X_train, Y_train, title="Training set")
boundary.show(model, X_validation, Y_validation_encoded, title="Validation set")
boundary.plot(history)
