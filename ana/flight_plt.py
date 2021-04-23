#!/usr/bin/env python
"""

::

    ipython -i flight_plt.py RoundaboutXY 
    ipython -i flight_plt.py RoundaboutXZ
    ipython -i flight_plt.py RoundaboutXY_XZ 

"""
import os, sys, numpy as np
from opticks.ana.makeflight import Flight

try:
    import pyvista as pv
except ImportError:
    pv = None
pass


def plot3d_arrows(pos, nrm, mag=1, grid=False):
    """ 
    """
    pl = pv.Plotter()
    pl.add_arrows(pos, nrm, mag=mag, color='#FFFFFF', point_size=2.0 )   
    if grid:
        pl.show_grid()
    pass
    cp = pl.show()
    return cp


if __name__ == '__main__':
    name = sys.argv[1]
    f = Flight.Load(name)
    plot3d_arrows( f.e[:,:3], f.g[:,:3], mag=0.5, grid=True )   
pass

