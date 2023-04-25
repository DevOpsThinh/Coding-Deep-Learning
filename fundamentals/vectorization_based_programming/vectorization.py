# Author: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
# Programming via Vectorization

import jax.numpy as jnp
import n1 as n1
import n2
import n3
import numpy as np
import timeit
import r3
from jax import jit


# Poor (inefficiently)
def clip_loops(x_matrix):
    for i in range(x_matrix.shape[0]):
        for j in range(x_matrix.shape[1]):
            if x_matrix[i, j] > 1:
                x_matrix[i, j] = 1
            elif x_matrix[j, j] < 0:
                x_matrix[i, j] = 0
    return x_matrix


def cov_loops(x_matrix):
    y_matrix = x_matrix.shape[0]
    d = x_matrix.shape[1]  # d dimensions

    mean = np.zeros(d)
    for i in range(y_matrix):
        for j in range(d):
            mean[j] += x_matrix[i, j]
    for j in range(d):
        mean[j] /= y_matrix

    sample_covariance_matrix = np.zeros((d, d))
    z = np.zeros(d)
    for i in range(y_matrix):
        for j in range(d):
            z[j] = x_matrix[i, j] - mean[j]
        for m in range(d):
            for n in range(d):
                sample_covariance_matrix[m, n] += z[m] * z[n]
    for m in range(d):
        for n in range(d):
            sample_covariance_matrix[m, n] /= y_matrix
    return sample_covariance_matrix


# Efficiently
# a_matrix = np.clip(a_matrix, 0, 1)

X = np.random.normal(size=(5000, 784)) * 1.5
X1 = clip_loops(X)
X2 = np.clip(X, 0, 1)  # (about 300 -400 times faster)

print(np.sum((X1 - X2) * (X1 - X2)))


timeit.timeit(clip_loops(X))
timeit.timeit(np.clip(X, 0, 1))


def cov_vec_cpu(x_matrix):
    mean = np.mean(a_matrix, axis=0)
    return (x_matrix - mean) @ (x_matrix - mean).T / x_matrix.shape[0]


@jit
def cov_jax_gpu(x_matrix):
    x_matrix = jnp.array(x_matrix)
    mean = jnp.mean(a_matrix, axis=0)
    return (x_matrix - mean) @ (x_matrix - mean).T / x_matrix.shape[0]


# Compare
a_matrix = np.random.normal(size=(300, 784))

# timeit.timeit(-n1, -r1, cov_loops(a_matrix))
# timeit.timeit(-n2, -r2, cov_vec_cpu(a_matrix))
# timeit.timeit(-n3, -r3, cov_jax_gpu(a_matrix))
b_matrix = cov_loops(a_matrix)
c_matrix = cov_vec_cpu(a_matrix)
d_matrix = cov_jax_gpu(a_matrix)
print(np.trace(b_matrix), np.trace(c_matrix), np.trace(d_matrix))

a_matrix = np.random.normal(size=(5000, 784))

timeit.timeit(cov_vec_cpu(a_matrix))
timeit.timeit(cov_jax_gpu(a_matrix))

c_matrix = cov_vec_cpu(a_matrix)
d_matrix = cov_jax_gpu(a_matrix)
print(np.trace(c_matrix), np.trace(d_matrix))

"""
    Summary:
    
    1. vectorization is about 20000+ times faster than loops;
    2. GPUs can further speed it up about 10 times;
    3. GPUs can accelerate more for larger matrices, about 150+ times faster at above.
"""
