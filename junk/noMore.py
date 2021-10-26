import numpy as np

A = np.array([1,2,3,4,1,2,8,7,3,2,1,4,5,2,4,])
b = np.array([], dtype=int)

for i in A:
    if i not in b:
        b = np.append(b, i)

print(b)