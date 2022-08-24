# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Mathematical Thinking in Computer Science
# Recursion & Induction:
#       Analyzing the correctness & running time of algorithms.

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
# T(i) = 2T(n - 1) + 1: T(7) = 2 x 63 + 1 = 127 || T(8) = 2 * 127 + 1 = 255
def hanoi_towers(i, from_rod, to_rod):
    if i == 1:
        print(f'Move disk from {from_rod} to {to_rod}')
    else:
        unused_rod = 6 - from_rod - to_rod
        hanoi_towers(i - 1, from_rod, unused_rod)
        print(f'Move disk from {from_rod} to {to_rod}')
        hanoi_towers(i - 1, unused_rod, to_rod)


hanoi_towers(3, 1, 2)
