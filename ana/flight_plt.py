#!/usr/bin/env python

import os, numpy as np
from opticks.ana.flight import Flight

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import mpl_toolkits.mplot3d.art3d as art3d

if __name__ == '__main__':

    plt.ion()
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111,projection='3d')
    plt.title("flight_plt")
    sz = 25

    ax.set_xlim([-sz,sz])
    ax.set_ylim([-sz,sz])
    ax.set_zlim([-sz,sz])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    f = Flight.Load()
    elu = f.elu
    #print(elu[:,3,:4].copy().view("|S2"))

    sc = 10 

    f.quiver_plot(ax, sc=sc)
    fig.show()
pass

