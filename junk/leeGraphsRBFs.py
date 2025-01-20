import numpy as np
import matplotlib.pyplot as plt

def plotLee(xx,invSys,subind,x,c):
    plotSoln = np.zeros(xx.size)
    for i in range(xx.size):
        sum = 0
        for j in range(x.size):
            sum += np.sqrt((xx[i]-x[j])**2+c**2) * invSys[subind-1,j]
        plotSoln[i] = sum
    return plotSoln

h = .1
x = np.arange(-.2,.2+h,h)
x2 = np.arange(0,.4+h,h)
c = .4

sys = np.zeros((x.size,x.size))

for i in range(x.size):
    for j in range(x.size):
        # if i == 0 or i == 4:
        #     sys[i,j] = np.sqrt((x[i]-x[j])**2+c**2)
        # else:
        #     sys[i,j] = (x[i] - x[j]) / np.sqrt((x[i]-x[j])**2+c**2)
        sys[i,j] = np.sqrt((x[i]-x[j])**2+c**2)

invSys = np.linalg.pinv(sys)

xx = np.arange(-.2,.2+.001,.001)
psi1 = plotLee(xx,invSys,1,x,c)
psi2 = plotLee(xx,invSys,2,x,c)
psi3 = plotLee(xx,invSys,3,x,c)
psi4 = plotLee(xx,invSys,4,x,c)
psi5 = plotLee(xx,invSys,5,x,c)
xxx = np.arange(0,.4+.001,.001)
psi11 = plotLee(xxx,invSys,1,x2,c)
psi21 = plotLee(xxx,invSys,2,x2,c)
psi31 = plotLee(xxx,invSys,3,x2,c)
psi41 = plotLee(xxx,invSys,4,x2,c)
psi51 = plotLee(xxx,invSys,5,x2,c)
u1 = 0
u2 = 1
u3 = 2
u4 = 1
u5 = 0
u6 = 2
u7 = 3

summy = psi1*u1 + psi2*u2 + psi3*u3 + psi4*u4 +psi5*u5
summy2 = psi11*u3 + psi21*u4 + psi31*u5 + psi41*u6 +psi51*u7

plt.plot(xx,summy,xxx,summy2)
plt.grid()
plt.show()