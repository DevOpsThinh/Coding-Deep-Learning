# Co-Author/ Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# The helper functions
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set the plotting DPI settings to be a bit higher.
plt.rcParams['figure.figsize'] = [8.0, 5.0]
plt.rcParams['figure.dpi'] = 150


def train_with_history(x, y, iterations, lr, precision, init_w, init_b):
    w, b = init_w, init_b

    previous_loss = loss(x, y, w, b)
    history = [[w, b, previous_loss]]
    for i in range(0, iterations):
        w_gradient, b_gradient = gradient_two_variables(x, y, w, b)
        w -= lr * w_gradient
        b -= lr * b_gradient

        c_loss = loss(x, y, w, b)
        history.append([w, b, previous_loss])

        if abs(c_loss - previous_loss) < precision:
            return w, b, history
        previous_loss = c_loss
    raise Exception("Couldn't converge within %d iterations" % iterations)


def gradient_two_variables(x, y, w, b):
    """ Calculate the gradient of the curve """
    w_gradient = 2 * np.average(x * (predict(x, w, b) - y))
    b_gradient = 2 * np.average(predict(x, w, b) - y)
    return w_gradient, b_gradient


def predict(x, w, b):
    return x * w + b


def loss(x, y, w, b):
    return np.average((predict(x, w, b) - y) ** 2)


def load_text_dataset(text_dataset):
    return np.loadtxt(text_dataset, skiprows=1, unpack=True)


def plot_the_chart(x, y, x_label, y_label, w, b):
    sns.set()
    plt.plot(x, y, "bo")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(x_label, fontsize=30)
    plt.ylabel(y_label, fontsize=30)
    x_edge, y_edge = 50, 50
    plt.axis([0, x_edge, 0, y_edge])
    plt.plot([0, x_edge], [b, predict(x_edge, w, b)], linewidth=1.0, color="y")
    plt.show()
