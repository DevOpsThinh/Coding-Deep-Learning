# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning with linear regression

from ml.util import load_text_dataset, loss, plot_the_chart, predict


def train(x, y, iterations, lr):
    wei = bias = 0
    for i in range(iterations):
        c_loss = loss(x, y, wei, bias)
        print("Iteration %4d => Loss: %.6f" % (i, c_loss))
        if loss(x, y, wei + lr, bias) < c_loss:
            wei += lr
        elif loss(x, y, wei - lr, bias) < c_loss:
            wei -= lr
        elif loss(x, y, wei, bias + lr) < c_loss:
            bias += lr
        elif loss(x, y, wei, bias - lr) < c_loss:
            bias -= lr
        else:
            return wei, bias
    raise Exception("Couldn't converge within %d iterations" % iterations)


# Import the pizza dataset
X, Y = load_text_dataset("../../fundamentals/datasets/pizza_forester/pizza.txt")
# Train phase: Train the system
w, b = train(X, Y, iterations=10000, lr=0.01)
print("\nw=%.3f, b=%.3f" % (w, b))
# Predict phase: Predict the number of pizzas
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))
# Plot the chart
plot_the_chart(X, Y, "Reservations", "Pizzas", w, b)
