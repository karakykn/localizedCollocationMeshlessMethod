import numpy as np

# np.random.seed(1)
# Psi = np.random.randint(0, high=10, size=[4,4])
# invPsi = np.linalg.inv(Psi)
#
# Psi_d = np.random.randint(0, high=10, size=[4,4])
# invPsi_d = np.linalg.inv(Psi_d)
#
# B = np.random.randint(0, high=10, size=4)
#
# print( np.matmul( Psi_d , np.matmul( invPsi , B ) ) )
#
# print( np.matmul( np.linalg.pinv( np.matmul( Psi , invPsi_d ) ) , B ) )

A = np.array([2,3,4])
B = np.array([3,3,2])
print(np.append(A,B))