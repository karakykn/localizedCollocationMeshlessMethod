import numpy as np
import matplotlib.pyplot as plt

h=1/40
x=np.arange(-1,1+h,h)
c=0.4
m=20

subDomainRadius = 2*h
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

bigSys = np.zeros((x.size,x.size))
rhs = np.zeros(x.size)

for k in range(subDomNo):
    for i in range(subDomains[k].shape[0]):
        if subDomains[k][i] != 0 and subDomains[k][i] != x.size-1:
            bigSys[subDomains[k][i], subDomains[k][:]] += np.matmul(fxx[k][i,:], invPnns[k])
            rhs[subDomains[k][i]] += 105/2 * x[subDomains[k][i]]**2 -7.5
        else:
            bigSys[subDomains[k][i], subDomains[k][:]] += np.matmul(f[k][i,:], invPnns[k])
            rhs[subDomains[k][i]] += 1


# rhs[1] -= bigSys[1,0] * rhs[0]
# rhs[2] -= bigSys[2,0] * rhs[0]
#
# rhs[-2] -= bigSys[-2,-1] * rhs[-1]
# rhs[-3] -= bigSys[-3,-1] * rhs[-1]
#
# bigSys = np.delete(bigSys, 0, 0)
# bigSys = np.delete(bigSys, 0, 1)
# bigSys = np.delete(bigSys, -1, 0)
# bigSys = np.delete(bigSys, -1, 1)
#
# rhs = np.delete(rhs, 0, 0)
# rhs = np.delete(rhs, -1, 0)

invBigSys = np.linalg.pinv(bigSys)

unk = np.matmul(invBigSys,rhs)
rel = 35/8*x**4 - 15/4*x**2 + 3/8
# unk = np.concatenate((np.array([rel[0]]), unk, np.array([rel[-1]])))
eta = 0
for i in range(rel.size):
    eta += (unk[i]-rel[i])**2
eta = eta / np.dot(rel,rel)
eta = np.sqrt(eta)*100
print('Eta: ', eta)
print('Av. Abs. err.: ', np.sum(np.abs(unk-rel))/rel.size)
print('max. abs. err: ', np.max(np.abs(unk-rel)))

plt.plot(x, unk, '--', color='k', linewidth=1, markersize=3)
plt.plot(x, rel, color='k', linewidth=1)
plt.show()
