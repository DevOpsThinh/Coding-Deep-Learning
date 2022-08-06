# This is a Python Crash Course script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from math import sqrt


def print_basic_arithmetic(result):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Result: {result}')  # Press Ctrl+F8 to toggle the breakpoint.


def tuple_distance(tuple1, tuple2):
    return sqrt((tuple1[0] - tuple2[0]) ** 2 + (tuple1[1] - tuple2[1]) ** 2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Basic Arithmetic:
    print_basic_arithmetic(((4 - 2) + 4) / 5 * 9)
    # Variable Assignment:
    x = (1, 0)
    y = (5, 3)
    print(f'Distance between {x} & {y} is {tuple_distance(x, y)}')
    # Strings
    s1 = '\tShe love you'
    s2 = '\nyeah \n\tyeah \n\t\tyeah'
    print(s1 + s2)
    my_age = 31
    print('My age is: {}'.format(my_age - 2))
    # Lists
    aList = [5, 6, 7, 8, 9, True, 'a lot of different type of things']
    print(aList)
    otherList = [1, 2, 3, 4] + aList
    print(otherList)
    print('The last item in otherList list is with "-1th" element: ', otherList[- 1])
    print('The last item in otherList list is with len() function: ', otherList[len(otherList) - 1])
    print(otherList[0:4])
    otherList[9] = False
    print(otherList)
    otherList.remove(1)
    print(otherList)
    otherList.append(True)
    print(otherList)
    # Sets operations: Union - Intersection - Difference
    A = {'Dog', 'Cat', 'Pig', 'Chicken', 'Rabbit', 'Turtle'}
    B = {'Dog', 'Chicken', 'Monkey', 'Cow'}
    U = A.union(B)
    print('The union of set A & set B is: ', U)
    N = A.intersection(B)
    print('The intersection of set A & set B is: ', N)
    D = A.difference(B)
    print('The difference of set A & set B is ', D)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
