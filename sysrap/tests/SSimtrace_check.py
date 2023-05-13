#!/usr/bin/env python

import os, numpy as np, matplotlib.pyplot as plt
SIZE = np.array([1280, 720])

s = np.load("simtrace.npy")
lpos = s[:,1,:3]

title = os.getcwd()

fig,ax = plt.subplots(figsize=SIZE/100)
ax.set_aspect('equal')
fig.suptitle(title)

ax.scatter( lpos[:,0], lpos[:,2] )
fig.show() 


