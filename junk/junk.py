import numpy as np

from LCMM.classes import *
# from Thesis.funcs.meshlessFuncs import *
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
import time as tm

a = np.array([0,1,2,3,4])
b = a
a[2] = 3
print(a,b)