#!/usr/bin/env python
import numpy as np

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import mpl_toolkits.mplot3d.art3d as art3d

path = "/tmp/FlightPathTest.npy"
elui = np.load(path).reshape(-1,4,4)


plt.ion()
fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111,projection='3d')
plt.title(path)

ax.plot( elui[:,0,0], elui[:,0,1], elui[:,0,2] )  


fig.show()







