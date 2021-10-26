import numpy as np
import matplotlib.pyplot as plt

h=1/40
x=np.arange(0,1+h,h)
c=1
dt = .0001
endTime = 1

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
alpha = np.array(np.zeros(subDomNo), dtype=object)
soln = np.array(np.zeros(subDomNo), dtype=object)
globalSoln = np.zeros(x.size)

globalSoln[0] = 1
globalSoln[-1] = 1

for k in range(subDomNo):
    subDomNodeNo = subDomains[k].shape[0]
    phiHat = np.zeros((subDomNodeNo,subDomNodeNo))
    phiHat_x = np.zeros((subDomNodeNo,subDomNodeNo))
    phiHat_xx = np.zeros((subDomNodeNo,subDomNodeNo))
    for i in range(subDomNodeNo):
        for j in range(subDomNodeNo):
            r_sq = (x[subDomains[k][i]] - x[subDomains[k][j]])**2
            r = np.sqrt(r_sq)
            phiHat[i,j] = np.sqrt( r_sq + c**2 )
            phiHat_x[i, j] = (x[subDomains[k][i]] - x[subDomains[k][j]]) / phiHat[i, j]
            phiHat_xx[i, j] = 1 / phiHat[i, j] - (x[subDomains[k][i]] - x[subDomains[k][j]]) ** 2 / phiHat[i, j] ** 3
            # if i!=j:
            #     phiHat[i,j] = r**m*np.log(r)
            #     phiHat_x[i, j] = (x[subDomains[k][i]] - x[subDomains[k][j]]) * r**(m-2) * (m*np.log(r)+1)
            #     phiHat_xx[i, j] = r**(m-2) * (m*np.log(r)+1) + (x[subDomains[k][i]] - x[subDomains[k][j]])**2 * r**(m-4) * (2*(m-1) + m*(m-2)*np.log(r))
    f[k] = phiHat
    fx[k] = phiHat_x
    fxx[k] = phiHat_xx
    invPnns[k] = np.linalg.pinv(f[k])

step = int(endTime/dt)
for time in range(step):

    for k in range(subDomNo):
        soln[k] = np.zeros(subDomains[k].shape[0])
        for i in range(subDomains[k].shape[0]):
            soln[k][i] = globalSoln[subDomains[k][i]]
        alpha[k] = np.matmul(invPnns[k],soln[k])
    for k in range(subDomNo):
        globalSoln[subDomains[k][1]] += dt*np.matmul(fxx[k][1,:],alpha[k])

    print(globalSoln)