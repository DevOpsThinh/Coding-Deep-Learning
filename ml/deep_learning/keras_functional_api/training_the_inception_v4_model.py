# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Deep Learning with Keras framework (A deep learning library)
#           Implementing Inception-v4 with the Keras Functional API
#           Training the Inception-v4 model

import os
import datetime
import pandas as pd
import math

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from inception_v4_network import model
from ml.deep_learning.keras_functional_api.helper_function import plot_training_stats

# ----------------- A computer without an NVIDIA GPU,
# Download it at: https://drive.google.com/file/d/1NqMs2js-uOyOLL9MB0nzX0AoHx-hy8ty/view?usp=sharing
# Place it on the following path: "ml/deep_learning/keras_functional_api/inceptionv4.dogscats_weights.h5"
# then uncomment the first two statements:

# weights_path = 'inceptionv4.dogscats_weights.h5'
# model.load_weights(weights_path, by_name=True)

# ----------------- A computer have an NVIDIA GPU
# Adjust these to match the dimensions of our input image.
IMAGE_HEIGHT = 299
IMAGE_WIDTH = 299
# Reduce this if this model does not fit on your GPU.
BATCH_SIZE = 24
# Load all file names in the /datasets/dogs-vs-cats/train directory into a dataframe
img_dir = '../../../fundamentals/datasets/dogs_vs_cats/train/'
# Enumerate all files in source directory & prepare an array that only contains
# file names that this code can support (.jpg, .png)
f_names_temp = next(os.walk(img_dir))[2]
f_names = []
obj_classes = []

for f in f_names_temp:
    name, extension = os.path.splitext(f)
    if extension.lower() == '.jpg':
        f_names.append(f)
        if name.startswith('cat'):
            obj_classes.append('0')
        elif name.startswith('dog'):
            obj_classes.append('1')

df_training_data = pd.DataFrame(list(zip(f_names, obj_classes)), columns=['Filename', 'ObjectClass'])
# df_training_data.head().info()

# Create train & test sets
x_train, x_test, y_train, y_test = train_test_split(df_training_data['Filename'], df_training_data['ObjectClass'],
                                                    test_size=0.25, random_state=42,
                                                    stratify=df_training_data['ObjectClass'])
num_train_samples = len(x_train)
num_test_samples = len(x_test)

df_train = pd.DataFrame(list(zip(x_train, y_train)), columns=['Filename', 'ObjectClass'])
df_test = pd.DataFrame(list(zip(x_test, y_test)), columns=['Filename', 'ObjectClass'])
# Confirmation when object instances of classes are destroyed.
del x_train, x_test, y_train, y_test
# Create Keras generator for training data
# Rescale all pixel values to the range 0.0 to 1.0
# Apply in-memory augmentations to original data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.75, 1.25],
    height_shift_range=0.2,
    width_shift_range=0.2,
    rotation_range=180.0,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=img_dir,
    x_col="Filename",
    y_col="ObjectClass",
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
# Create Keras generator for testing data
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=img_dir,
    x_col="Filename",
    y_col="ObjectClass",
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

start_time = datetime.datetime.now()
call_backs = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('inception_v4_checkpoint_weights.h5', verbose=1, save_best_only=True, save_weights_only=True)
]
# Train the model for 30 epochs
history = model.fit(train_generator, steps_per_epoch=math.ceil(num_train_samples / BATCH_SIZE),
                    epochs=30, callbacks=call_backs, validation_data=test_generator,
                    validation_steps=math.ceil(num_test_samples / BATCH_SIZE))

plot_training_stats(history)

model.save('./inception_v4_dogs_cats_full_model.mod')
model.save_weights('inception_v4_dogs_cats_weights.h5')

end_time = datetime.datetime.now()
print(f'training completed in: {end_time - start_time}')

"""
=> Attention! (Note):
The Inception-v4 model was trained on a Windows 10 computer workstation with 64GB of RAM 
and an NVIDIA RTX 2080Ti 11Gb graphics card. It took about three hours to train 30 epochs, 
with the best model having an accuracy of 87.9 percent.
"""