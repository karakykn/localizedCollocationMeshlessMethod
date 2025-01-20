from LCMM.classes import *
import matplotlib.pyplot as plt
import pandas as pd

# """Input variables for meshing"""
# spatialSteps = [2.5,3.5]
# vertices = np.array([[0,-7],[40,-7],[40,7],[0,7],[0,-7]]) #counter clockwise domain vertices, start from lower left point
#
# """Meshing..."""
# print("Meshing...\n")
# mesh = Mesh(spatialSteps=spatialSteps, vertices=vertices) #Mesh defined here
#
# df = pd.DataFrame(np.array([mesh.locations[:,0], mesh.locations[:,1]]).T, columns=['x','y'])
# df.to_csv('records/mesher.csv', index=False)

th = np.linspace(0,2*np.pi,80)
x = np.cos(th) + 8
y = np.sin(th)

df = pd.DataFrame(np.array([x, y]).T, columns=['x','y'])
df.to_csv('records/mesher.csv', index=False)