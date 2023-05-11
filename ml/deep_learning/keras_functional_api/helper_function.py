# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Deep Learning with Keras framework (A deep learning library)
#           A collection of utilities functions

from keras import regularizers
from keras.initializers import initializers_v1
from keras import backend as be

from keras.models import Model
from keras.layers.convolutional import MaxPooling2D, Conv2D, AveragePooling2D
from keras.layers import Input, Dropout, Dense, Flatten, Activation
from keras.layers.merging import concatenate
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.optimizers import Adam

# Hyperparameters we can adjust
DROPOUT_PROBABILITY = 0.1
INITIAL_LEARNING_RATE = 0.001
L2_REGULARIZATION_AMOUNT = 0.00004

# Adjust these to match the dimensions of our input image.
IMAGE_HEIGHT = 299
IMAGE_WIDTH = 299
IMAGE_CHANNELS = 3

# Reduce this if this model does not fit on our GPU.
BATCH_SIZE = 24


def build_reduction_b_block(input_tensor):
    """
    A reduction block: Transform a 35x35 input into a 17x17 input in an efficient manner.
    :param input_tensor: The input image tensor
    :return: outputs of the three input branches
    """
    # This is the first branch from the left
    branch_left = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input_tensor)
    # This is the middle branch
    branch_middle = conv2d_batch_norm_relu(input_tensor, 192 , 1, 1)
    branch_middle = conv2d_batch_norm_relu(branch_middle, 192, 3, 3, strides=(2, 2), padding='valid')
    # This is the right branch
    branch_right = conv2d_batch_norm_relu(input_tensor, 256, 1, 1)
    branch_right = conv2d_batch_norm_relu(branch_right, 256, 1, 7)
    branch_right = conv2d_batch_norm_relu(branch_right, 320, 7, 1)
    branch_right = conv2d_batch_norm_relu(branch_right, 320, 3, 3, strides=(2, 2), padding='valid')
    # Concatenate all the results from the three branches
    outputs = concatenate([branch_left, branch_middle, branch_right], axis=-1)
    return outputs


def build_reduction_a_block(input_tensor):
    """
    A reduction block: Transform a 35x35 input into a 17x17 input in an efficient manner.
    :param input_tensor: The input image tensor
    :return: outputs of the three input branches
    """
    # This is the first branch from the left
    branch_left = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input_tensor)
    # This is the middle branch
    branch_middle = conv2d_batch_norm_relu(input_tensor, 384, 3, 3, strides=(2, 2), padding='valid')
    # This is the right branch
    branch_right = conv2d_batch_norm_relu(input_tensor, 192, 1, 1)
    branch_right = conv2d_batch_norm_relu(branch_right, 224, 3, 3)
    branch_right = conv2d_batch_norm_relu(branch_right, 256, 3, 3, strides=(2, 2), padding='valid')
    # Concatenate all the results from the three branches
    outputs = concatenate([branch_left, branch_middle, branch_right], axis=-1)
    return outputs


def build_inception_c_block(input_tensor):
    """
    Create the Inception C block - an Inception-v4 block
    :param input_tensor: The input image tensor
    :return: outputs of the four input branches
    """
    # (384 1x1 convolutions) - This is the first branch for the left
    branch_a = conv2d_batch_norm_relu(input_tensor, 256, 1, 1)
    # This is the second branch for the left
    branch_b = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input_tensor)
    branch_b = conv2d_batch_norm_relu(branch_b, 256, 1, 1)
    # This  is the third branch from the left
    branch_c = conv2d_batch_norm_relu(input_tensor, 384, 1, 1)
    branch_c_left = conv2d_batch_norm_relu(branch_c, 256, 1, 3)
    branch_c_right = conv2d_batch_norm_relu(branch_c, 256, 3, 1)
    # This is the fourth (right-most) branch
    branch_d = conv2d_batch_norm_relu(input_tensor, 384, 1, 1)
    branch_d = conv2d_batch_norm_relu(branch_d, 448, 1, 3)
    branch_d = conv2d_batch_norm_relu(branch_d, 512, 3, 1)
    branch_d_left = conv2d_batch_norm_relu(branch_d, 256, 1, 3)
    branch_d_right = conv2d_batch_norm_relu(branch_d, 256, 3, 1)
    # Concatenate all the results from the four branches
    outputs = concatenate([branch_a, branch_b, branch_c_left, branch_c_right, branch_d_left, branch_d_right], axis=-1)
    return outputs


def build_inception_b_block(input_tensor):
    """
    Create the Inception B block - an Inception-v4 block
    :param input_tensor: The input image tensor
    :return: outputs of the four input branches
    """
    # (384 1x1 convolutions) - This is the first branch for the left
    branch_a = conv2d_batch_norm_relu(input_tensor, 384, 1, 1)
    # This is the second branch for the left
    branch_b = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input_tensor)
    branch_b = conv2d_batch_norm_relu(branch_b, 128, 1, 1)
    # This  is the third branch from the left
    branch_c = conv2d_batch_norm_relu(input_tensor, 192, 1, 1)
    branch_c = conv2d_batch_norm_relu(branch_c, 224, 1, 7)
    branch_c = conv2d_batch_norm_relu(branch_c, 256, 7, 1)
    # This is the fourth (right-most) branch
    branch_d = conv2d_batch_norm_relu(input_tensor, 192, 1, 1)
    branch_d = conv2d_batch_norm_relu(branch_d, 192, 1, 7)
    branch_d = conv2d_batch_norm_relu(branch_d, 224, 7, 1)
    branch_d = conv2d_batch_norm_relu(branch_d, 224, 1, 7)
    branch_d = conv2d_batch_norm_relu(branch_d, 256, 7, 1)
    # Concatenate all the results from the four branches
    outputs = concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)
    return outputs


def build_inception_a_block(input_tensor):
    """
    Create the Inception A block - an Inception-v4 block
    :param input_tensor: The input image tensor
    :return: outputs of the four input branches
    """
    # (96 1x1 convolutions) - This is the first branch for the left
    branch_a = conv2d_batch_norm_relu(input_tensor, 96, 1, 1)
    # This is the second branch for the left
    branch_b = conv2d_batch_norm_relu(input_tensor, 64, 1, 1)
    branch_b = conv2d_batch_norm_relu(branch_b, 96, 3, 3)
    # This  is the third branch from the left
    branch_c = conv2d_batch_norm_relu(input_tensor, 64, 1, 1)
    branch_c = conv2d_batch_norm_relu(branch_c, 96, 3, 3)
    branch_c = conv2d_batch_norm_relu(branch_c, 96, 3, 3)
    # This is the fourth (right-most) branch
    branch_d = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input_tensor)
    branch_d = conv2d_batch_norm_relu(branch_d, 96, 1, 1)
    # Concatenate all the results from the four branches
    outputs = concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)
    return outputs


def conv2d_batch_norm_relu(input_tensor, num_kernels, kernel_rows, kernel_cols, padding='same', strides=(1, 1)):
    """
    Create a 2D convolutional layer.
    Apply batch normalization to the output of the convolutional layer, and then apply a rectified linear unit
    activation function to the normalization output.
    :param input_tensor: The input image tensor
    :param num_kernels: Convolutional kernels
    :param kernel_rows: height dimension
    :param kernel_cols: width dimension
    :param padding: one of `"valid"` or `"same"` (case-insensitive). `"valid"` means no padding. `"same"`
    results in padding with zeros evenly to the left/right or up/down of the input. When `padding="same"` and
    `strides=1`, the output has the same size as the input.
    :param strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height
    and width.
    :return: The normalization output of the 2D convolutional layer
    """
    x = Conv2D(num_kernels, (kernel_rows, kernel_cols), strides=strides, padding=padding, use_bias=False,
               kernel_regularizer=regularizers.l2(L2_REGULARIZATION_AMOUNT),
               kernel_initializer=initializers_v1._v1_glorot_normal_initializer(seed=42))(input_tensor)
    x = BatchNormalization()(x)

    output = Activation('relu')(x)

    return output


