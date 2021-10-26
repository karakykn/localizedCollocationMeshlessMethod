from LCMM.classes import *
import matplotlib.pyplot as plt
import scipy.spatial
import matplotlib.tri as mtri


"""Input variables for meshing"""
spatialSteps = [1/4,1/4]
vertices1 = np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]]) #counter clockwise domain vertices, start from lower left point
rbf = 'MQ'
shapeParameter = 16
dt = 1e-4
endTime = 1e0
iterLimit = 3e3
mode = 'transient'

"""Meshing..."""
print("Meshing...\n")
mesh = Mesh(spatialSteps=spatialSteps, vertices=vertices1) #Mesh defined here
mesh.interface(2/3, shape='cercle', intBoundary_dtheta=np.pi/8)

beta1 = 1
beta2 = 2
k1 = 0
k2 = 0
g2 = 10
#
solution = LRBFCM(mesh, shapeParam=shapeParameter, rbf=rbf)
solution.interfacePoissonMixTest(mode, g2, dt=dt, endTime=endTime, tolerance=1e-6, iterLim=iterLimit, beta2=beta2, beta1=beta1, k1=k1, k2=k2)

df = pd.DataFrame(np.array([solution.solnOuter, solution.mesh.locations[:,0], solution.mesh.locations[:,1]]).T, columns=['Approx. soln.','x','y'])
df.to_csv('records/results_intefraceOuter.csv', index=False)

df = pd.DataFrame(np.array([solution.solnInner, solution.mesh.interfaceLocs[:,0], solution.mesh.interfaceLocs[:,1]]).T, columns=['Approx. soln.','x','y'])
df.to_csv('records/results_intefraceInner.csv', index=False)

# """Plotting the nodes"""
# fig = plt.figure()
# ay = fig.gca()
# ay.plot(solution.mesh.locations[:,0],solution.mesh.locations[:,1],'x',color='blue')
# for i in range(solution.mesh.locations.shape[0]):
#     ay.text(solution.mesh.locations[i,0]+.01, solution.mesh.locations[i,1]-.05, str(i),color='blue')
# ay.plot(solution.mesh.interfaceLocs[:,0],solution.mesh.interfaceLocs[:,1],'o',color='red')
# for i in range(solution.mesh.interfaceLocs.shape[0]):
#     ay.text(solution.mesh.interfaceLocs[i,0]-.01, solution.mesh.interfaceLocs[i,1]+.05, str(i),color='red')

# """Plotting the approximated solution"""
# fig1 = plt.figure()
# ax = fig1.gca(projection='3d')
# tess = scipy.spatial.Delaunay(solution.mesh.locations)
# x = tess.points[:, 0]
# y = tess.points[:, 1]
# tri = tess.vertices # or tess.simplices depending on scipy version
# triang = mtri.Triangulation(x=solution.mesh.locations[:, 0], y=solution.mesh.locations[:, 1], triangles=tri)
# ax.plot_trisurf(triang, solution.solnOuter)
# plt.xlabel('x')
# plt.ylabel('y')
# ax.set_zlim(0,2)
# plt.show()
