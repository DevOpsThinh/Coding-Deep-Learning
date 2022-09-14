# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Mathematical Thinking in Computer Science
# Mathematical Logic: Negation (!), Logical AND - Conjunction (/\),
# Logical OR - Disjunction (\/), If-Then (Implication) (==>)
#
# Ref: https://en.wikipedia.org/wiki/Negation

from itertools import combinations

from fundamentals.discrete_mathematics.algorithms.logic import *

# Does there exist a power of 2 that starts with 65: 2^16 = 65536
for x in range(100):
    if int(str(2 ** x)[:2]) == 65:
        print(f'2^{x} = {2 ** x}')

# n > 1, n^2 + n + 41 is prime
for x in range(1, 100):
    if not is_prime_long_version(x ** 2 + x + 41):
        print(x)
        # exit()

print((next(x for x in range(2, 100) if not is_prime_short_version(x * x + x + 41))))

# Do there exist positive integers a, b, c such that a^2 + b^2 = c^2?
# The Pythagorean theorem - Triangles
for a, b, c in combinations(range(1, 20), 3):
    if a ** 2 + b ** 2 == c ** 2:
        print(f'{a}^2 + {b}^2 = {c}^2')

a_list = [5, 17, 6, 10]
print(not any([is_divisible_by_3(i) for i in a_list]))  # False
print(all([not is_divisible_by_3(i) for i in a_list]))  # False

"""
    Summary
    
    1. One example is enough to prove an existential statement;
    2. One counterexample is enough to refute a universal statement;
    3. !, /\, \/, ==> are basic logical operations;
    4. The negation - not(!) of a universal statement is an existential statement, and vice versa;
    5. Proof by contradiction (or reductio ad absurdum) is a basic argument: to prove that
        a statement is true, one assumes that its opposite is true & derives a contradiction;
    6. Reductio ad absurdum is on of the most popular proof arguments, is usually combined with other proof ideas.
"""
