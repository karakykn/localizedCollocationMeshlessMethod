import numpy as np

A = np.array([[1,1,0],[0,1,0],[-1,1,0]])
B = np.array([[1,1],[0,1],[-1,1]])

invA = np.linalg.pinv(A)
invB = np.linalg.pinv(B)

print(B)
print(invB)

print(np.matmul(B,invB))
print(np.matmul(invB,B))