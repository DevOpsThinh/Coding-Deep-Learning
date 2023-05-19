# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Deep Learning: Convolutional Neural Networks (CNNs)
#           The Keras's CIFAR-10 images dataset
#           Prints random sample images this dataset.

from keras.datasets import cifar10
import matplotlib.pyplot as plt
import random

# Load CIFAR-10
(X, Y), (_, _) = cifar10.load_data()

# Print a 4x10 grid of images
ROWS = 4
COLUMNS = 10
for i in range(ROWS * COLUMNS):
    ax = plt.subplot(ROWS, COLUMNS, i + 1)  # Get the next cell in the grid
    ax.axis('off')                          # Remove ticks on axes
    idx = random.randint(0, X.shape[0])     # Select a random image
    ax.set_title(Y[idx][0], fontsize=15)    # Print the image's label
    ax.imshow(X[idx])                       # Show the image
plt.show()