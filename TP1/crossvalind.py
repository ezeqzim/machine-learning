import random
a = [_ for _ in range(0, 45000)]
random.shuffle(a)

b = [0 for _ in range(0, len(a))]

for i in range (0, len(a)/10):
  b[a[i]] += 1

print b
