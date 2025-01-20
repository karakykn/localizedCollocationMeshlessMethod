import numpy as np
import matplotlib.pyplot as plt

h=1/2
x=np.arange(-1,1+h,h)
c=0.4

subDomainRadius = h
subDomNo = x.size-2
epsilon=1e-14
subDomains = np.array(np.zeros(subDomNo),dtype=object)
jj = 0
for i in range(1,x.size-1):
    xi = x[i]
    influencer = np.array([], dtype=int)
    for j in range(x.size):
        xj = x[j]
        distance = np.abs(xi-xj)
        if distance<=subDomainRadius+epsilon:
            influencer = np.append(influencer, j)
    subDomains[jj] = influencer
    jj += 1

f = np.array(np.zeros(subDomNo),dtype=object)
fx = np.array(np.zeros(subDomNo),dtype=object)
fxx = np.array(np.zeros(subDomNo),dtype=object)
fy = np.array(np.zeros(subDomNo),dtype=object)
fyy = np.array(np.zeros(subDomNo),dtype=object)
invPnns = np.array(np.zeros(subDomNo), dtype=object)

for k in range(subDomNo):
    subDomNodeNo = subDomains[k].shape[0]
    phiHat = np.zeros((subDomNodeNo,subDomNodeNo))
    phiHat_x = np.zeros((subDomNodeNo,subDomNodeNo))
    phiHat_y = np.zeros((subDomNodeNo,subDomNodeNo))
    phiHat_xx = np.zeros((subDomNodeNo,subDomNodeNo))
    phiHat_yy = np.zeros((subDomNodeNo,subDomNodeNo))
    for i in range(subDomNodeNo):
        for j in range(subDomNodeNo):
            r_sq = (x[subDomains[k][i]] - x[subDomains[k][j]])**2
            phiHat[i,j] = np.sqrt( r_sq + c**2 )
            phiHat_x[i, j] = (x[subDomains[k][i]] - x[subDomains[k][j]]) / phiHat[i, j]
            phiHat_xx[i, j] = 1 / phiHat[i, j] - (x[subDomains[k][i]] - x[subDomains[k][j]]) ** 2 / phiHat[i, j] ** 3
    f[k] = phiHat
    fx[k] = phiHat_x
    fxx[k] = phiHat_xx
    invPnns[k] = np.linalg.pinv(f[k])

invLocalSys = np.array(np.zeros(subDomNo), dtype=object)
for k in range(subDomNo):
    if k != 0 and k != subDomNo-1:
        invLocalSys[k] = np.linalg.pinv(np.matmul(fxx[k],invPnns[k]))
    elif k == 0:
        dummy = fxx[k]
        dummy[0,:] = f[k][0,:]
        invLocalSys[k] = np.linalg.pinv(np.matmul(dummy,invPnns[k]))
    else:
        dummy = fxx[k]
        dummy[2,:] = f[k][2,:]
        invLocalSys[k] = np.linalg.pinv(np.matmul(dummy,invPnns[k]))

rhsGlobal = np.zeros(x.size)
rhsLocal = np.zeros(3)
solnLocal = np.zeros(3)
soln = np.zeros(x.size)
soln[0], soln[-1] = 1, 1
dummySoln = np.zeros(x.size)

dite = .001
tolerance = 1e-4
residual = 1
while residual > tolerance:
    dummySoln[:] = soln[:]
    for k in range(subDomNodeNo):
        if k==0:
            rhsLocal[0] = soln[subDomains[k][0]]
            rhsLocal[1] = rhsGlobal[subDomains[k][1]]
            rhsLocal[2] = rhsGlobal[subDomains[k][2]]
        elif k == subDomNodeNo-1:
            rhsLocal[0] = rhsGlobal[subDomains[k][0]]
            rhsLocal[1] = rhsGlobal[subDomains[k][1]]
            rhsLocal[2] = soln[subDomains[k][2]]
        else:
            for i in range(3):
                rhsLocal[i] = rhsGlobal[subDomains[k][i]]
        soln[subDomains[k][1]] = dite  * np.matmul(invLocalSys[k][1,:],rhsLocal) + dummySoln[subDomains[k][1]]

    residual = np.max(np.abs(dummySoln-soln))
    print(residual)
    print(soln)

print('\n')
print('soln: ', soln)