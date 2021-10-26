from LCMM.classes import *
import matplotlib.pyplot as plt

"""Convection kısmı hatalı"""

"""Input variables for meshing"""
spatialSteps = [1/8,1/8]
vertices = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]]) #counter clockwise domain vertices, start from lower left point
dt = 1e-4
endTime = 2.5e-1

"""Meshing..."""
print("Meshing...\n")
mesh = Mesh(spatialSteps=spatialSteps, vertices=vertices) #Mesh defined here

# """Input variables for RBFs and the problem"""
rbf = 'MQ'
shapeParameter = 16
# shapeParameter = 2
boundaryConditions = np.array([1, 1, 1, 1])
# 0 for Dirichlet, 1 for Neumann, 2 for Robin Boundary Condition
# Do not forget to define source and boundary functions in "classes.py"
velocityField = np.zeros((mesh.nodeNo,2))
for i in range(mesh.nodeNo):
    velocityField[i,0] = 1

"""Solving..."""
print("Solving...\n")
solution = LRBFCM(mesh, shapeParam=shapeParameter, rbf=rbf, boundaries=boundaryConditions)
solution.scalarTransport(velocityField=velocityField, dt=dt,endTime=endTime, thermalConductivity=0)
soln = solution.soln

df = pd.DataFrame(np.array([soln, solution.mesh.locations[:,0], solution.mesh.locations[:,1]]).T, columns=['Approx. soln.','x','y'])
df.to_csv('records/results.csv', index=False)

"""Plotting the approximated solution"""
plt.tricontourf(solution.mesh.locations[:,0], solution.mesh.locations[:,1], soln, antialiased=True, cmap='seismic')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.show()