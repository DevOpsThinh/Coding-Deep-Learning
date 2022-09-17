# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Mathematical Thinking in Computer Science
#
# Basic Counting

from itertools import product, permutations

from networkx import DiGraph, all_simple_paths

# Sum rule: Listing all integers from 1 to 10 that are divisible by either 2 or 3
print([i for i in range(1, 11) if i % 2 == 0 or i % 3 == 0])

# Sets
a = {5, 2, 8, 17, 2}
b = {3, 17, 2, 19, 6, 17}

print(f'Duplicates are removed automatically:\n'
      f'a = {a}\n'
      f'b = {b}\n')

print(f'Size of {a} is {len(a)}\n'
      f'Size of {b} is {len(b)}\n')

print(f'2 belongs to {a}: {2 in a}\n'
      f'5 belongs to {a}: {3 in a}\n'
      f'2 belongs to {b}: {2 in b}\n'
      f'5 belongs to {b}: {5 in b}\n')

print(f'Union of {a} and {b}: {a.union(b)}\n'
      f'Intersection of {a} and {b}: {a.intersection(b)}\n')

print(f'Set building:\n'
      f'Set of odd numbers of {a} is {set(i for i in a if i % 2 == 1)}\n'
      f'Set of numbers from {b} that don\'t belong to {a}: '
      f'{set(j for j in b if j not in a)}')

# Recursive Counting
# Listing all paths from s to t in the s-t network
edges = [('s', 'a'), ('s', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'e'), ('c', 'd'),
         ('c', 't'), ('d', 't'), ('e', 't')]

graph = DiGraph(edges)
for path in all_simple_paths(graph, source='s', target='t'):
    print(*path, sep='->')

# agraph = nx_agraph.to_agraph(graph)
# agraph.graph_attr.update(rankdir='LR')
# agraph.draw('st_network.png', prog='dot')
# Image('st_network.png')

# Product Rule
for letter in {'e', 'u'}:
    for digit in {8, 2, 5}:
        print(letter, digit)

for i in product({'e', 'u'}, {8, 2, 5}):
    print(*i)

# Tuples and Permutations
# Enumerating all passwords consisting of two lower case Latin letters
for number, password in enumerate(
        product('abcdefghijklmnopqrstuvwxyz', repeat=2)):
    print(number, ''.join(password), end=' ')

print()

# Counting integers from 0 to 9999 having exactly one digit 7.
print(len([i for i in range(10000) if str(i).count('7') == 1]))
print(len([j for j in product(range(10), repeat=4) if j.count(7) == 1]))

# Counting integers from 0 to 9999 having at least one digit 7.
print(len([k for k in range(10000) if '7' in str(k)]))

for q in permutations('abc', 2):
    print(*q)
