# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Mathematical Thinking in Computer Science
# Proofs: Convincing Arguments section

from random import randint, seed

# Number theory: a / b = k, if a = kb
print(237 % 3, "\t", 237 % 7)
# Is there a positive integer that is divisible by 13 and ends with 15?
for i in range(10 ** 4):
    if i % 13 == 0 and i % 100 == 15:
        print(i)
# Find a two-digit (positive) integer that becomes 7 times smaller when its first
# (=leftmost) digit is removed.
for i in range(10, 100):
    if i == 7 * int(str(i)[1:]):
        print(i)
# Non-constructive Proofs (probabilistic method) / Irrational numbers: x = a / b (a > 0, b > 0)
# 2^n + 3^n <= 10^1000
for i in range(4000):
    if 2 ** i + 3 ** i > 10 ** 1000:
        print(i - 1)
        break
# Finding a constructive proof: generate 30 seven-digit numbers.
seed(10)
for i in range(30):
    print(randint(10 ** 6, 10 ** 7 - 1), end=' ')
    if i % 6 == 5:
        print()
