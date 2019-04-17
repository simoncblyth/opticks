#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)
import numpy as np, math 

import matplotlib.pyplot as plt
from opticks.ana.torus_hyperboloid import Tor, Hyp

from opticks.ana.x018 import x018
from opticks.ana.x019 import x019
from opticks.ana.x020 import x020
from opticks.ana.x021 import x021


if __name__ == '__main__':
    plt.ion()
    fig = plt.figure(figsize=(6,5.5))


    plt.title("xplt")

    x_018 = x018()
    x_019 = x019()
    x_020 = x020()
    x_021 = x021()

    y_018 = x_018.spawn_rationalized() 
    y_019 = x_019.spawn_rationalized() 
    y_020 = x_020.spawn_rationalized() 
    y_021 = x_021.spawn_rationalized() 

    xx = [x_018, x_019, x_020, x_021 ]
    yy = [y_018, y_019, y_020, y_021 ]
    #yy = [y_020]

    ax = fig.add_subplot(111)
    ax.set_ylim([-350,200])
    ax.set_xlim([-300,300])

    for x in yy:
        for pt in x.root.patches():
            print "pt ", pt
            ax.add_patch(pt)
        pass
    pass 

    fig.show()



"""
    ax.scatter( p[0], p[1] , marker="*")
    ax.scatter( -p[0], p[1] , marker="*" )

    ax.plot( [R, p[0]], [z0, p[1]] )
    ax.plot( [-R, -p[0]], [z0, p[1]] )

    ax.plot( [R-r, p[0]], [z0, p[1]] )
    ax.plot( [-R+r, -p[0]], [z0, p[1]] )

    ax.plot(  tr, tz )
    ax.plot( -tr, tz )

    ax.plot(  hr, tz , linestyle="dashed")
    ax.plot( -hr, tz , linestyle="dashed")
"""
