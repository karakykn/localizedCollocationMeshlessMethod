import numpy as np
from LCMM.classes import *
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time as tm
from csv import writer

"""c is optimized here"""

"""Input variables for meshing"""
spatialSteps = [1/10,1/10]
vertices = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]]) #counter clockwise domain vertices, start from lower left point

"""Meshing..."""
print("Meshing...\n")
mesh = Mesh(spatialSteps=spatialSteps, vertices=vertices) #Mesh defined here
# plt.plot(mesh.locations[:,0],mesh.locations[:,1],'o')
# plt.show()

# """Input variables for RBFs and the problem"""
rbf = 'MQ'
boundaryConditions = np.array([0, 0, 0, 0])
# 0 for Dirichlet, 1 for Neumann Boundary Condition
# Do not forget to define source and boundary functions in "classes.py"
relSolwB = np.zeros(mesh.nodeNo)
for i in range(mesh.nodeNo):
    relSolwB[i] = unknown(mesh.locations[i, 0], mesh.locations[i, 1])
relSol = relSolwB[mesh.interior]

optSpace=21
c = np.linspace(0.1,10.1,optSpace)
rmss=np.zeros(optSpace)

tolerance =5
for tol in range(tolerance):
    print(tol)
    for opt in range(optSpace):
        solution = LCMM(mesh, shapeParam=c[opt], rbf=rbf, boundaries=boundaryConditions)
        solution.steadyPoisson()
        solution.steadySolve()
        soln = solution.soln
        solnInt = soln[-mesh.interior.size:]

        rms = 0
        for i in range(relSol.size):
            rms += (solnInt[i]-relSol[i])**2
        rms = rms/relSol.size
        rmss[opt] = np.sqrt(rms)

    argmin=np.argmin(rmss)
    cc=c[argmin]
    if tol!=tolerance-1:
        if argmin==0:
            c[optSpace-1]=c[1]
        elif argmin==optSpace-1:
            c[0]=c[argmin-1]
        else:
            c[0]=c[argmin-1]
            c[optSpace-1]=c[argmin+1]
        cSpace=(c[optSpace-1]-c[0])/(optSpace-1)
        for i in range(1,optSpace-1):
            c[i]=c[0]+i*cSpace

print('Error...\n')
avAbsErr = np.sum(np.abs(relSol-solnInt))/relSol.size
maxAbsErr = np.max(np.abs(relSol-solnInt))
condNo = np.linalg.cond(solution.system)

with open('records/steadyPoisRes_'+rbf+'_N'+str(mesh.nodeNo)+'_c'+str(cc)+'.csv','a', newline='') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(['nodeNo','rbf','shapeParameter','avAbsErr','maxAbsErr','rms','condNo'])
    writer_object.writerow([mesh.nodeNo,rbf,cc,avAbsErr,maxAbsErr,rmss[argmin],condNo])
with open('records/steadyPoisList_'+rbf+'_N'+str(mesh.nodeNo)+'_c'+str(cc)+'.csv','a', newline='') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(['x','y','exactU','approxU'])
for i in range(mesh.nodeNo):
    with open('records/steadyPoisList_'+rbf+'_N'+str(mesh.nodeNo)+'_c'+str(cc)+'.csv','a', newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow([mesh.locations[i,0],mesh.locations[i,1],relSolwB[i],soln[i]])