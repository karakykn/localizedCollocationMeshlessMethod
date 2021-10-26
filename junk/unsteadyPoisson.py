import numpy as np
from LCMM.classes import *
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time as tm
from csv import writer

"""Input variables for meshing"""
spatialSteps = [.1,.1]
vertices = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]]) #counter clockwise domain vertices, start from lower left point
dt = 5e-4
endTime = 1e-3

"""Meshing..."""
print("Meshing...\n")
mesh = Mesh(spatialSteps=spatialSteps, vertices=vertices) #Mesh defined here

# """Input variables for RBFs and the problem"""
rbf = 'MQ'
shapeParameter = 8
# shapeParameter = 8
boundaryConditions = np.array([1, 0, 0, 1])
# 0 for Dirichlet, 1 for Neumann, 2 for Robin Boundary Condition
# Do not forget to define source and boundary functions in "classes.py"

"""Solving..."""
print("Solving...\n")
solution = LCMM(mesh, shapeParam=shapeParameter, rbf=rbf, boundaries=boundaryConditions)
solution.unsteadyPoisson(dt, materialCoefficient=1)
solution.unsteadySolve(endTime)
soln = solution.soln
solnInt = soln[-solution.mesh.interior.size:]
#print(solnInt,'\n')

# print(soln)
df = pd.DataFrame(np.array([soln, solution.mesh.locations[:,0], solution.mesh.locations[:,1]]).T, columns=['Approx. soln.','x','y'])
df.to_csv('records/sarlerEx2.csv', index=False)

##################################
# relSolwB = np.zeros(mesh.nodeNo)
# for i in range(mesh.nodeNo):
#     relSolwB[i] = unknown(solution.mesh.locations[i, 0], solution.mesh.locations[i, 1], endTime)
# relSol = relSolwB[solution.mesh.interior]
# print(relSol,'\n')
# print('Error...\n')
# rms = 0
# for i in range(relSolwB.size):
#     rms += (soln[i]-relSolwB[i])**2
# rms = rms/relSolwB.size
# rms = np.sqrt(rms)
# print('RMS: ', rms)
# avAbsErr = np.sum(np.abs(relSolwB-soln))/relSolwB.size
# maxAbsErr = np.max(np.abs(relSolwB-soln))
# condNo = np.linalg.cond(solution.system)
# print('Abs Err:', avAbsErr)
# print('Max. abs. err: ', maxAbsErr)
# eta = 0
# for i in range(relSolwB.size):
#     eta += (soln[i]-relSolwB[i])**2
# eta = eta / np.dot(relSolwB,relSolwB)
# eta = np.sqrt(eta)*100
# print('Eta: ', eta)