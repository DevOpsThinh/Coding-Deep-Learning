# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Mathematical Thinking in Computer Science
# Recursion & Induction:
#       Analyzing the correctness & running time of algorithms.

from math import log2

import matplotlib.pyplot as plt
import numpy as np
from numpy import prod


# Counting people
def length(a_list):
    print(f'Computing the length of {a_list}...')
    if not a_list:  # Alternatively, replace this with: a_list == []
        print(f'The length of {a_list} is 0')
        return 0
    else:
        print(f'The length of {a_list} is {1 + length(a_list[1:])}')
        return 1 + length(a_list[1:])


print(length([5, 3, 2, 1, 7, 9]))


def length_opt(a_list):
    return 1 + length(a_list[1:]) if a_list else 0


print(length_opt(['thinh', 'thang', 'thien']))


# n!
def factorial(n):
    assert n > 0
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


# n! = (n - 1)! x n
def factorial_opt(n):
    assert n > 0
    if n == 1:
        return 1
    else:
        return n * factorial_opt(n - 1)


print(factorial(9))
print(factorial_opt(9))


def factorial_with_zero_func(n):
    assert n >= 0
    return prod(range(1, n + 1))


def factorial_with_zero_control_flow(n):
    assert n >= 0
    return n * factorial(n - 1) if n else 1


print(factorial_with_zero_func(9))
print(factorial_with_zero_control_flow(9))


# Recuse with Care
# n >= 8, n = 3 + 5 + 3 + 5 + ... || n = 3 + 3 +, ... || n = 5 + 5 + ...
def with_draw(amount):
    assert amount >= 8
    if amount == 8:
        return [3, 5]
    if amount == 9:
        return [3, 3, 3]
    if amount == 10:
        return [5, 5]

    coins = with_draw(amount - 3)
    coins.append(3)
    return coins


coins_amount = 101
x_coin = with_draw(coins_amount)
print(f"{coins_amount} = {' + '.join(map(str, x_coin))}")


# The compact solution
def with_draw_base_cases(amount, base_cases):
    assert amount >= 8
    if amount <= 10:
        return base_cases[amount]
    return with_draw_base_cases(amount - 3) + [3]


base_cases_dict = {8: [3, 5], 9: [3, 3, 3], 10: [5, 5]}
print(with_draw_base_cases(9, base_cases_dict))


# Non-recursive program
def with_draw_non_recursive(amount):
    assert amount >= 8
    coins = []
    while amount % 3 != 0:
        coins.append(5)
        amount -= 5
    while amount != 0:
        coins.append(3)
        amount -= 3
    return coins


print(with_draw_non_recursive(13))


# The classical Hanoi Towers puzzle
# T(i) = 2T(i - 1) + 1: T(7) = 2 x 63 + 1 = 127 || T(8) = 2 * 127 + 1 = 255
def hanoi_towers(i, from_rod, to_rod):
    if i == 1:
        print(f'Move disk from {from_rod} to {to_rod}')
    else:
        unused_rod = 6 - from_rod - to_rod
        hanoi_towers(i - 1, from_rod, unused_rod)
        print(f'Move disk from {from_rod} to {to_rod}')
        hanoi_towers(i - 1, unused_rod, to_rod)


hanoi_towers(3, 1, 2)


# Binary Search - Slowly growing function: log(2)n = b if 2^b = n.
# Guess a number problem
def query(y):
    x = 1618235
    if x == y:
        return 'equal'
    elif x < y:
        return 'smaller'
    else:
        return 'greater'


def guess(lower, upper):
    middle = (lower + upper) // 2
    answer = query(middle)
    print(f'Is x = {middle}? It is {answer}.')
    if answer == 'equal':
        return
    elif answer == 'smaller':
        guess(lower, middle - 1)
    else:
        assert answer == 'greater'
        guess(middle + 1, upper)


# Guess an integer 1<= x <= 2097151
guess(1, 2097151)


def divide_till_one(r):
    divisions = 0
    while r > 1:
        r = r // 2
        divisions += 1
    return divisions


# If n <= 10^9 => log(2)n < 30
for i in range(1, 10):
    n = 10 ** i
    print(f'{n} {log2(n)} {divide_till_one(n)}')

# "Linear scan" with python
print(5 in [1, 3, 4, 11, 5, 10, 7, 8, 2, 9])
print(6 in [1, 3, 4, 11, 5, 10, 7, 8, 2, 9])


def binary_search(a, x):
    print(f'Searching {x} in {a}')

    if len(a) == 0:
        return False
    if a[len(a) // 2] == x:
        print('Found!')
        return True
    elif a[len(a) // 2] < x:
        return binary_search(a[len(a) // 2 + 1:], x)
    else:
        return binary_search(a[:len(a) // 2], x)


binary_search([1, 3, 5, 11, 4, 10, 7, 8, 2, 9], 5)
binary_search([1, 2, 3, 4, 5, 7, 8, 9, 10, 11], 5)

# Induction
# n >= 1, the sum of integers from 1 to n is n(n+1) / 2
for i in (17, 251, 1356):
    assert sum(range(1, i + 1)) == i * (i + 1) // 2

# Compound Percents
# Bernoulli's inequality: x >= -1 and n >= 0: (1 + x)^n >= 1 + xn

plt.xlabel('$n$')
plt.ylabel('Money (vnd)')

x = np.arange(250)
plt.plot(x, 1.02 ** x, label='$1.02^n$')
plt.plot(x, 1 + 0.02 * x, label='1+0.2^n')
plt.legend(loc='upper left')
plt.savefig('bernoulli.png')
plt.show()
