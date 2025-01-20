from LCMM.classes import *

spatialSteps = [1/3,1/3]
vertices = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])

print("Meshing...\n")
mesh = Mesh(spatialSteps=spatialSteps, vertices=vertices)

rbf = 'MQ'
shapeParameter = 8
dt = .001
endTime = 1
boundaryConditions = np.array([0, 0, 0, 0])

solution = leeMQ(mesh, shapeParam=shapeParameter, rbf=rbf, boundaries=boundaryConditions)

soln = np.ones(solution.mesh.locations.shape[0])
soln[-4:] = 0
print(soln)
step = int(endTime/dt)
alpha = np.array(np.zeros(solution.subDomNo), dtype=object)
localSoln = np.array(np.zeros(solution.subDomNo), dtype=object)
for time in range(step):

    for k in range(solution.subDomNo):
        localSoln[k] = np.zeros(solution.subDomains[k].shape[0])
        for i in range(solution.subDomains[k].shape[0]):
            localSoln[k][i] = soln[solution.subDomains[k][i]]
        alpha[k] = np.matmul(solution.invPnns[k],localSoln[k])

    soln[solution.subDomains[0][2]] += dt*np.matmul(solution.fyy[0][2,:]+solution.fxx[0][2,:],alpha[0])
    soln[solution.subDomains[1][3]] += dt*np.matmul(solution.fyy[1][3,:]+solution.fxx[1][3,:],alpha[1])
    soln[solution.subDomains[2][3]] += dt*np.matmul(solution.fyy[2][3,:]+solution.fxx[2][3,:],alpha[2])
    soln[solution.subDomains[3][4]] += dt*np.matmul(solution.fyy[3][4,:]+solution.fxx[3][4,:],alpha[3])

    print(soln)