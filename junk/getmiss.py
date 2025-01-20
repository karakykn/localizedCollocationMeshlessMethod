import numpy as np

def getMissing(upLimit, array):
    holder = np.array([],dtype=int)
    for i in range(upLimit):
        if i not in array:
            holder = np.append(holder, i)
    return holder


a = np.array([0,1,2,4,5,2,4,7])
print(getMissing(7,a))