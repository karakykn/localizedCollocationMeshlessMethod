from LCMM.classes import *
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time as tm

"""Input variables for meshing"""
spatialSteps = [1/2,1/2]
vertices = np.array([[0,0],[1.5,0],[1.5,1],[0,1],[0,0]])  #counter clockwise domain vertices, start from lower left point

"""Meshing..."""
print("Meshing...")
mesh = Mesh(spatialSteps=spatialSteps, vertices=vertices) #Mesh defined here

plt.plot(mesh.locations[:,0],mesh.locations[:,1],'x')
plt.show()