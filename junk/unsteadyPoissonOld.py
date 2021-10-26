from LCMM.junk.classesOld import *
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time as tm

"""Input variables for meshing"""
spatialSteps = [.25,.25]
vertices = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]]) #counter clockwise domain vertices, start from lower left point

"""Meshing..."""
print("Meshing...")
mesh = Mesh(spatialSteps=spatialSteps, vertices=vertices) #Mesh defined here

"""Input variables for RBFs and the problem"""
rbf = 'MQ'
shapeParameter = 4*spatialSteps[0]
boundaryConditions = np.array([0, 0, 0, 0])
# 0 for Dirichlet, 1 for Neumann Boundary Conditions
# Do not forget to define source and boundary functions in "classes.py"

"""Solving..."""
print("Solving...")
solution = LCMM(mesh, shapeParam=shapeParameter, rbf=rbf, boundaries=boundaryConditions)
solution.unsteadyPoisson(timeStep=1e-2)
solution.unsteadyPoissonSolve(100e-2)
soln = solution.soln
print(soln)