# Author: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
# Programming objects:
#           Encapsulating data
#           Addressing class attributes
#           Built-in attributes (on fundamentals/custom_functions.py)
#           Collecting garbage
#           Inheriting features
#           Overriding base methods
#           Harnessing polymorphism

class Polygon:
    """A base class to define Polygon properties"""
    width = 0
    height = 0
    count = 0

    def __int__(self, name_of_polygon):
        """
         Initializer function
        :param name_of_polygon: Name of the polygon
        :return: None
        """
        self.name = name_of_polygon

    def who_am_i(self):
        print('\nI\'m a polygon')

    def set_values(self, width, height):
        Polygon.width = width
        Polygon.height = height
        Polygon.count += 1

    def scale(self, number=1):
        print(self.name, ': Calling the base class -', number)

    def __del__(self):
        """
        A destructor method for confirmation when Polygon instances of
        class are destroyed.
        :return: None
        """
        print(f'{self.name} Say Goodbye!')


class Rectangle(Polygon):
    """ A derived class that inherits from Polygon class"""
    def area(self):
        return self.width * self.height

    def scale(self, number):
        self.width = self.width * number
        self.height = self.height * number

    def who_am_i(self):
        print('\nI\'m a rectangles')


class Triangle(Polygon):
    """ A derived class that inherits from Polygon class"""
    def area(self):
        return (self.width * self.height) / 2

    def scale(self, number):
        self.width = self.width * number

    def who_am_i(self):
        print('\nI\'m a triangles')
