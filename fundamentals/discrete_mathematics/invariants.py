# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Mathematical Thinking in Computer Science
#
# Invariants: used in analyzing the behavior of algorithms, programs, and other processes.

from fundamentals.discrete_mathematics.algorithms.logic import sort_books, represent

# Thinh is debugging his code. When he starts, he has only one bug. But once he fixes
# one bug, three new bugs appear. Last night Thinh fixed 15 bugs. How many pending
# bugs does Thinh have now?

number_of_pending_bugs = 1
for _ in range(15):
    number_of_pending_bugs -= 1
    number_of_pending_bugs += 3

print(f'Now, Thinh has {number_of_pending_bugs} pending bugs')

# Arthur's Books
sort_books([4, 7, 2, 3, 8, 9, 0, 1, 6, 5])

# Even & Odd numbers
for i in (7, 15, 22, 33):
    print(i, represent(9, i))

"""
    Summary
    
    1. Invariants are important tools for proving impossibility, termination, & various bounds.
    2. Invariants may take many forms: numbers, "being even or odd", equations, inequalities.
    2. Double counting is a method that uses the sum invariant.
"""
