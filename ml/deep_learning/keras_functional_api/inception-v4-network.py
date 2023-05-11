# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Deep Learning with Keras framework (A deep learning library)
#           Implementing Inception-v4 with the Keras Functional API

from keras import backend as be

# Check the data ordering format (if we're using Theano as the backend => should be channels_first.).
be.set_image_data_format('channels_last')
print(be.image_data_format())
