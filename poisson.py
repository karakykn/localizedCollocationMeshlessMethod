from classes import *
import matplotlib.pyplot as plt

"""Input variables for meshing"""
spatialSteps = [1/10,1/10]
vertices = np.array([[0,0],[10,0],[10,2],[0,2],[0,0]]) #counter clockwise domain vertices, start from lower left point
dt = 1e-4
endTime = 1e-1
iterLimit = 3e3
mode = 'transient' # or 'transient'

"""Meshing..."""
print("Meshing...\n")
mesh = Mesh(spatialSteps=spatialSteps, vertices=vertices) #Mesh defined here

# """Input variables for RBFs and the problem"""
rbf = 'MQ'
shapeParameter = 16
# shapeParameter = 2
boundaryConditions = np.array([0, 1, 0, 0])
# 0 for Dirichlet, 1 for Neumann, 2 for Robin Boundary Condition
# Do not forget to define source and boundary functions in "classes.py"

"""Solving..."""
print("Solving...\n")
solution = LRBFCM(mesh, shapeParam=shapeParameter, rbf=rbf, boundaries=boundaryConditions)
solution.poisson(mode=mode,dt=dt,endTime=endTime,iterLim =iterLimit, materialCoefficient=1)
soln = solution.soln

df = pd.DataFrame(np.array([soln, solution.mesh.locations[:,0], solution.mesh.locations[:,1]]).T, columns=['Approx. soln.','x','y'])
df.to_csv('records/results.csv', index=False)

"""Plotting the approximated solution"""
fig1 = plt.figure()
ax = fig1.gca(projection='3d')
ax.plot_trisurf(solution.mesh.locations[:,0], solution.mesh.locations[:,1], soln, antialiased=True, linewidth=0.4)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""Plotting the approximated solution"""
# plt.tricontourf(solution.mesh.locations[:,0], solution.mesh.locations[:,1], soln, antialiased=True, cmap='seismic')
# plt.colorbar()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()