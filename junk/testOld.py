from LCMM.junk.classesOld import *
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time as tm

"""Input variables for meshing"""
spatialSteps = [1/2,1/2]
vertices = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]]) #counter clockwise domain vertices, start from lower left point

"""Meshing..."""
print("Meshing...")
mesh = Mesh(spatialSteps=spatialSteps, vertices=vertices) #Mesh defined here

"""Input variables for RBFs and the problem"""
rbf = 'MQ'
shapeParameter = spatialSteps[0]/mesh.nodeNo
boundaryConditions = np.array([0, 0, 0, 0])
# 0 for Dirichlet, 1 for Neumann Boundary Condition
# Do not forget to define source and boundary functions in "classes.py"

"""Solving..."""
print("Solving...")
solution = LCMM(mesh, shapeParam=shapeParameter, rbf=rbf, boundaries=boundaryConditions)
solution.steadyPoisson()
solution.steadySolve()
soln = solution.soln
print(soln)
print(np.linalg.cond(solution.system))

#kassab iterative deneme
# invLocalRBF = np.linalg.pinv(solution.localRBF)
# B = solution.rhs
# L = np.matmul(solution.system, invLocalRBF)
# for i in range(200):
#     yeni = np.matmul(L, B)
#     for j in (mesh.interior):
#         B[j] += yeni[j]
# print(B[mesh.interior])