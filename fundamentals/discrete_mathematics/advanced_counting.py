# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Mathematical Thinking in Computer Science
#
# Combinations with Repetitions - Combinatorics & Probability

from itertools import combinations, combinations_with_replacement, product, permutations

from fundamentals.discrete_mathematics.algorithms.logic import print_it

# Tuples (also known as k-words - n^k) are ordered selections with repetitions.
for p in product('abc', repeat=2):
    print_it(p)
print()
# Permutations (also known as k-permutations - $ \ frac{n!}{(n - k)!}) are ordered selections without repetitions.
for k in permutations('abc', 2):
    print_it(k)
print()
# Combinations (also known as k-sets - {n! / k!(n - k)!}) are unordered selections without repetitions.
for i in combinations('abc', 2):
    print_it(i)
print()
# Combinations with repetitions are unordered selections with repetitions (also known as k-multisets).
for j in combinations_with_replacement('abc', 2):
    print_it(j)
print()
# Problem: We have an unlimited supply of tomatoes, bell peppers, & lettuce. We would like to make
# a salad out of four units among three ingredients (we don't have to use all ingredients).
# How many different salads can we make?
for salad in combinations_with_replacement('TBL', 4):
    print(*salad)
print()

for bars_indices in combinations(range(6), 2):
    sequence = ['*'] * 6
    for i in bars_indices:
        sequence[i] = '|'
    print(*sequence, ' ', *bars_indices)
