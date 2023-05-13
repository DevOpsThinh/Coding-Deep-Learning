# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Deep Learning with Keras framework (A deep learning library)
#           Implementing Inception-v4 with the Keras Functional API
#           Create an InceptionV4 network - An Inception Architecture.

from keras import backend as be

from keras.models import Model
from keras.layers.convolutional import AveragePooling2D
from keras.layers import Input, Dropout, Dense, Flatten
from keras.optimizers import Adam

from helper_function import build_inception_v4_conv_base

# Hyperparameters we can adjust
DROPOUT_PROBABILITY = 0.1
INITIAL_LEARNING_RATE = 0.001

# Adjust these to match the dimensions of our input image.
IMAGE_HEIGHT = 299
IMAGE_WIDTH = 299
IMAGE_CHANNELS = 3

# Check the data ordering format (if we're using Theano as the backend => should be channels_first.).
be.set_image_data_format('channels_last')
print(be.image_data_format())

# The previous layer of the network (or the input image tensor).
INPUT_TENSOR = Input((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
# Convolutions Base
conv_base = build_inception_v4_conv_base(INPUT_TENSOR)
# The classifier on top of the inception-v4 convolutions base.
pool_output = AveragePooling2D((8, 8), padding='valid')(conv_base)
dropout_output = Dropout(DROPOUT_PROBABILITY)(pool_output)
flattened_output = Flatten()(dropout_output)
network_output = Dense(units=2, activation='softmax')(flattened_output)
# Use the Adam optimizer and compile the model.
adam_opt = Adam(lr=INITIAL_LEARNING_RATE)
model = Model(INPUT_TENSOR, network_output, name='InceptionV4')
model.compile(optimizer=adam_opt, loss='binary_crossentropy', metrics=["accuracy"])
# Display a summary of the layers of the model.
model.summary()
# ===========================================
#   Total params: 41,209,058
#   Trainable params: 41,145,890
#   Non-trainable params: 63,168
# _______________________________________________
