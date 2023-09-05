#!/usr/bin/env python

import os, numpy as np
import matplotlib.pyplot as mp  
SIZE = np.array([1280, 720])

cdat = os.path.expandvars("$CDAT")
a = np.loadtxt(cdat, delimiter="f,      //"  ) 

tab0 = a[:250]
tab1 = a[250:]


fig, ax = mp.subplots(figsize=SIZE/100.)

ax.axvline( 2e-6 )  
ax.axvline( 5e-4 )  


ax.scatter( a[:,1], a[:,0], s=0.1 )
#ax.scatter( tab0[:,1], tab0[:,0], s=0.1 )
#ax.scatter( tab1[:,1], tab1[:,0], s=0.1 )



fig.show()







