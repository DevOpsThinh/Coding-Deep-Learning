# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Mathematical Thinking in Computer Science
#
# Binomial Coefficients - Combinatorics & Probability

from itertools import combinations, product
from math import comb


# Problem 1: Number of games in a Tournament
# Five teams played a tournament: each team played once with each other.
# What was the number of games?
def f(n):
    """
    Recursive counting
    :param n: The number of teams played the tournament.
    :return: The number of teams
    """
    return (n - 1) + f(n - 1) if n > 1 else 0


print(f(5))

# Enumerating the games
for game in combinations('abcde', 2):
    print(*game)
print()
# Problem 2: I'm organizing a car journey. I've five friends, but there are only
# three vacant places in my car. What is the number of ways of taking
# three of my five friends to the journey?

# Enumerating the groups
for group in combinations('abcde', 3):
    print(*group)
# Computing the number of groups
print(comb(5, 3))
# Computing binomial coefficients recursively
cells_dict = dict()  # cells_dict[i, j] will keep n choose k

for i in range(8):
    cells_dict[i, 0] = 1
    cells_dict[i, i] = 1
    for j in range(1, i):
        cells_dict[i, j] = cells_dict[i - 1, j - 1] + cells_dict[i - 1, j]

print(cells_dict[7, 4])
print()
# Practice Counting
# Problem 3: What is the number of four-digit integers whose digits are decreasing?
print(len([p for p in product(range(10), repeat=4)
           if p[0] > p[1] > p[2] > p[3]]))
