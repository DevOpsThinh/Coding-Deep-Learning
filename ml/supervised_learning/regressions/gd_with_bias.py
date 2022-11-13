# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning with linear regression - Gradient descent algorithm
from ml.util import loss, gradient_two_variables, load_text_dataset, predict


def train(x, y, iterations, lr):
    wei = bias = 0
    for i in range(iterations):
        print("Iteration %4d => Loss: %.10f" % (i, loss(x, y, wei, bias)))
        w_gradient, b_gradient = gradient_two_variables(x, y, wei, bias)
        wei -= w_gradient * lr
        bias -= b_gradient * lr
    return wei, bias


# Import the dataset
X, Y = load_text_dataset("../../../fundamentals/datasets/pizza_forester/pizza.txt")
# Training phase: train the system
w, b = train(X, Y, iterations=20000, lr=0.001)
print("\nw=%.10f, b=%.10f" % (w, b))
# Predict phase: predict the number of pizzas
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))
