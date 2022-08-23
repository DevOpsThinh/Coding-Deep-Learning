# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Mathematical Thinking in Computer Science
# Finding an example - Brute Force Search:
#       Narrowing the search space
#       Using known results
#       Identify more properties
#       Solve a Smaller Problem

from itertools import permutations, combinations

# Does there exist a six-digit integer that starts with 100 & is a multiple of 9127
# 100000 <= 9127k <= 100999 <=> 10.95
for i in range(100000, 101000):
    if i % 9127 == 0:
        print(i)
# Does there exist a three-digit integer that has remainder 1 when divided by 1, 2, 4, 5, 6, 7?
# N - 1 = (3 x 4 x 5 x 7)k = 420k <=> N = 420k + 1
for i in range(100, 1000):
    if all(i % j == 1 for j in range(2, 8)):
        print(i)

# Brute Force - Permutation:
# Step 1: Enumerate all possible placements of n queens where no two queens stay in the same row or column.
for p in permutations(range(4)):
    print(p)


# Step 2: Among all such placements, we select one where no two queens stay on the same diagonal.
def is_solution(perms):
    pairs = combinations(range(len(perms)), 2)
    return all(abs(i1 - i2) != abs(perms[i1] - perms[i2])
               for i1, i2 in pairs)


print(is_solution([1, 3, 0, 2]))
print(is_solution([2, 0, 3, 1]))
print(is_solution([3, 1, 0, 2]))


def is_solution_n_queens_problem(perms):
    pairs = combinations(range(len(perms)), 2)
    return all(abs(i1 - i2) != abs(perms[i1] - perms[i2])
               for i1, i2 in pairs)


solution = next(filter(is_solution_n_queens_problem, permutations(range(8))))
print(solution)


# Backtracking - Recursive
def extend(perms, i):
    if len(perms) == i:
        print(f'Final permutation {perms}')
        return
    print(f'Extending partial permutation {perms}...')
    for j in range(i):
        if j not in perms:
            extend(perms + [j], i)


extend(perms=[], i=3)
