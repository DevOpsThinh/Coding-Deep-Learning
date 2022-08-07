# Author: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
from math import sqrt


class Point:
    """
    A class for 2D point objects in Cartesian coordinate system.
    """
    def __init__(self, x, y):
        """
        Initializer function.
        :param x: The x coordinate of the specific point
        :param y: The Y coordinate of the specific point
        :return: None
        """
        self.x = x
        self.y = y

    def __str__(self):
        """
        The customized method for better string representation.
        :return: Point(x, y) string representation
        """
        return f'Point ({self.x}, {self.y})'

    def __add__(self, other):
        """
        To perform coordinate-wise additions
        :param other: Another point
        :return: A new point which correspond to the coordinates
        """
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """
        To perform coordinate-wise subtractions
        :param other: Another point
        :return: A new point which correspond to the coordinates
        """
        return Point(self.x - other.x, self.y - other.y)

    def distance(self, other):
        """
        Computes the Cartesian distance between the calling point
        and another point objects
        :param other: Another point
        :return: The Cartesian distance
        """
        dif = self - other
        return sqrt(dif.x ** 2 + dif.y ** 2)

    def __del__(self):
        """
        A destructor method for confirmation when Point instances of
        class are destroyed.
        :return: None
        """
        print(f'{self} Say Goodbye!')






