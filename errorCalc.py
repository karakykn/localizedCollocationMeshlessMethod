from LCMM.classes import *

def analyticalSarler2(x,y,t):
    relSoln_x = np.zeros(x.size)
    relSoln_y = np.zeros(x.size)
    for n in range(0,1000):
        for i in range(x.size):
            relSoln_x[i] += 4/np.pi * (-1)**n/(2*n+1) * np.exp(-(2*n+1)**2*np.pi**2*t/4) * np.cos((2*n+1)*np.pi*(x[i]-1)/2)
            relSoln_y[i] += 4/np.pi * (-1)**n/(2*n+1) * np.exp(-(2*n+1)**2*np.pi**2*t/4) * np.cos((2*n+1)*np.pi*(y[i]-1)/2)
    relSoln = relSoln_x * relSoln_y
    return relSoln

data = pd.read_csv('records/results.csv')
soln = data['Approx. soln.'].to_numpy()
x = data['x'].to_numpy()
y = data['y'].to_numpy()
t = 1e-1
c = 16

# relSoln = unknown(x,y,t)
relSoln = analyticalSarler2(x,y,t)
# print(relSoln)
# print(soln)

delT_max = np.max(np.abs(soln-relSoln))
ind = np.where(np.abs(soln-relSoln)==delT_max)
delT_avg = np.mean(np.abs(soln-relSoln))
# rms = rootMeanSquare(soln,relSoln)
df = pd.read_csv('records/sarler2006table8.csv')
df2 = pd.DataFrame({'t':[t], 'c':[c],'delT_avg':[delT_avg], 'delT_max':[delT_max],'p_max_x':x[ind], 'p_max_y':y[ind]})
df = df.append(df2)

# df = pd.DataFrame(columns=['t','c','delT_avg','delT_max','p_max_x','p_max_y'])
df.to_csv('records/sarler2006table8.csv', index=False)


print('Maximum error: ', delT_max)
print('Average error: ', delT_avg)
# print('RMSE: ', rms)

data.insert(1, 'Exact soln.', relSoln)
data.to_csv('records/results.csv', index=False)