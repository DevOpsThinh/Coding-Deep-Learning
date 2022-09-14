# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Mathematical Thinking in Computer Science

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
