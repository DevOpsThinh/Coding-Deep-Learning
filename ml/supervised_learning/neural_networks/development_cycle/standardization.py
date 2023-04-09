# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Neural Network in Practice
#           An example of standardized data.

import numpy as np

from ml.util import standardize_inputs

inputs = [1, 2, 3, 4, 5, 6, 8, -12]

print("Inputs: ", np.random.shuffle(inputs))
print("Inputs average: ", np.average(inputs))
print("Inputs standard deviation: ", np.std(inputs))

s_inputs = standardize_inputs(inputs)

# The standardized inputs average might not be exactly zero,
# because of rounding errors.
print("Standardized inputs: ", np.random.shuffle(s_inputs))
print("Standardized inputs average: ", np.average(s_inputs))
print("Standardized inputs standard deviation: ", np.std(s_inputs))


