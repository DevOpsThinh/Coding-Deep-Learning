# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Mathematical Thinking in Computer Science


def represent(n, value):
    """
    Tries to represent the given value by placing signs in the expression 1 +- 2 +- ... +- n.
    """
    if n == 0 and value == 0:
        return []

    total = sum(range(1, n + 1))

    if abs(value) > total or (total - value) % 2 != 0:
        return False

    if value >= 0:
        return represent(n - 1, value - n) + [n]
    else:
        return represent(n - 1, value + n) + [-n]


def sort_books(books):
    day = 0
    for i in range(len(books)):
        j = books.index(i)
        if j != i:
            books[i], books[j] = books[j], books[i]
            print(f'After day {day}: {books}')
            day += 1


def is_divisible_by_3(n):
    """
    Checks whether the given positive integer n is (or is not) a multiple of 3.
    :param n: A positive integer > 0
    :return:  A boolean value
    """
    return n % 3 == 0


def is_prime_short_version(n):
    """
    Checks whether the given positive integer n is prime number
    :param n: A positive integer > 1
    :return:  A boolean value
    """
    return n != 1 and all(n % i != 0 for i in range(2, n))


def is_prime_long_version(n):
    """
    Checks whether the given positive integer n is prime number
    :param n: A positive integer > 0
    :return: A boolean value
    """
    assert n > 0
    if n == 1:
        return False

    for i in range(2, n):
        if n % i == 0:
            return False

    return True
