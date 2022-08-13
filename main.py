# This is a Python Crash Course script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# Use a breakpoint in the code line below to debug your script.
from fundamentals.custom_functions import *
from object_oriented_mindset.cartesian_system_coordinate import Point
from object_oriented_mindset.polygon import *
from object_oriented_mindset.util_cartesian_system_coordinate import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Basic Arithmetic:
    print_basic_arithmetic()
    print_basic_arithmetic(((4 - 2) + 4) / 5 * 9)
    # Variable Assignment:
    x = (1, 0)  # Tuple: stores multiple fixed values in a sequence
    y = (5, 3)
    print(f'Distance between {x} & {y} is {tuple_distance(x, y)}')
    # Strings
    s1 = '\tShe love you'
    s2 = '\nyeah \n\tyeah \n\t\tyeah'
    print(s1 + s2)
    my_age = 31
    print('My age is: {}'.format(my_age - 2))
    # Lists (stores multiple values in an ordered index)
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
    # Set (stores multiple unique values in an unordered collection)
    # Sets operations: Union - Intersection - Difference
    A = {'Dog', 'Cat', 'Pig', 'Chicken', 'Rabbit', 'Turtle'}
    B = {'Dog', 'Chicken', 'Monkey', 'Cow'}
    U = A.union(B)
    print('The union of set A & set B is: ', U)
    N = A.intersection(B)
    print('The intersection of set A & set B is: ', N)
    D = A.difference(B)
    print('The difference of set A & set B is ', D)
    # Dictionary (stores multiple unordered key:value pairs)
    info_dict_dev = {'name': 'DevOpsThinh', 'majors': 'Mobile Engineer', 'age': '29'}
    info_dict_tester = {'name': 'NgocMai', 'majors': 'Tester', 'age': '24'}
    info_dict_po = {'name': 'VanToan', 'majors': 'Product Owner', 'age': '32'}
    info_dict_boss = {'name': 'Thang', 'majors': 'Director', 'age': '35'}
    print('Info: ', info_dict_dev)
    print('Is There A specialized Key?: ', 'specialized' in info_dict_dev)
    # Logic & Loops
    for i in range(0, 10):
        boolIsGreaterThanThree = (i >= 3)
        if i == 5:
            print('Ignores at 5, continues at i = 6')
            continue
        if i == 9:
            print('Over flow at i = 9')
            break
        print(f'{i} - Spam: {boolIsGreaterThanThree}')

    for i in range(0, 30):
        my_step_by_steps = (i % 3)
        if my_step_by_steps == 0:
            print(i)

    cube_nums = [i ** 3 for i in range(10)]
    print(cube_nums)

    a_number = 20
    while a_number > 10:
        a_number -= 3
        if a_number < 10:
            break
        print("value: ", a_number)
    # Custom functions
    friends_stock_list = [info_dict_dev, info_dict_tester, info_dict_po, info_dict_boss]
    data = get_data_from_list(friends_stock_list)
    print(data)
    # File operations
    file_name = 'dreariness.txt'
    content = 'She love me!\n'
    content += 'Yeah\n'
    content += '\tYeah\n'
    content += '\t\tYeah\n'
    write_a_file(file_name, 'w', content)
    read_a_file(file_name, 'r')
    update_a_file_with_block(file_name, 'r+', content, 35)
    re_write_a_file(file_name)
    file_with_pickle_data('pickle.dat')
    # Objects
    p_x = Point(1, 0)
    p_y = Point(5, 3)
    draw(p_x.x, p_x.y)
    draw(p_y.x, p_y.y)
    print(f'\nDistance between {p_x} & {p_y} is {p_x.distance(p_y)}')
    del p_x
    del p_y

    rect = Rectangle()
    rect.name = 'Rectangles'
    indentify(rect)
    rect.set_values(4, 5)
    print("Rectangle Area: ", rect.area())
    rect.scale(2)
    Polygon.scale(rect, 2)
    print("Rectangle Area: ", rect.area())

    tria = Triangle()
    tria.name = 'Triangles'
    indentify(tria)
    tria.set_values(4, 5)
    print("Triangle Area: ", tria.area())
    tria.scale(2)
    Polygon.scale(tria, 2)
    print("Triangle Area: ", tria.area())

    print('The number of polygons: ', tria.count)

    addressing_class_instance_attributes(tria)
    exam_built_in_attributes(tria)
    exam_built_in_class_dictionary(Triangle)

    del rect
    del tria

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
