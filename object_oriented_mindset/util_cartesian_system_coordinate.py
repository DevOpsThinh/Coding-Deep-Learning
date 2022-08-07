# Author: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
from matplotlib import pyplot as plt


def draw_arrow(axis):
    left, right = axis.get_xlim()
    bottom, top = axis.get_ylim()
    plt.arrow(
        left, 0, right - left, 0, length_includes_head=True, head_width=0.15
    )
    plt.arrow(
        0, bottom, 0, top - bottom, length_includes_head=True, head_width=0.15
    )


def draw(x, y):
    # set up range of the plot
    limit = max(x, y) + 1

    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.set_aspect('equal')
    # lines corresponding to x- and y- coordinates
    plt.plot([x, x], [0, y], '-', c='blue', linewidth=3)
    plt.plot([0, x], [y, y], '-', c='blue', linewidth=3)
    # actual point
    plt.scatter(x, y, s=100, marker='o', c='red')

    axis.set_xlim((-limit, limit))
    axis.set_ylim((-limit, limit))
    # axis arrows
    draw_arrow(axis)
    plt.grid()
    plt.show()
