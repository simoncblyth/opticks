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


class PVPlot3D(object):
    def __init__(self, *args, **kwa):
        self.pl = pv.Plotter(*args, **kwa)

    def add_grid(self):
        self.pl.show_grid()

    def gaze_up_right_arrows(self, pos, gaze, up, rhs, mag=0.5):
        self.pl.add_arrows( pos, gaze, mag=mag*1.0,  color='#FF0000', point_size=2.0 ) 
        self.pl.add_arrows( pos, up  , mag=mag*0.50, color='#00FF00', point_size=2.0 ) 
        self.pl.add_arrows( pos, rhs , mag=mag*0.25, color='#0000FF', point_size=2.0 ) 

    def mvp(self):
        """
        https://github.com/pyvista/pyvista-support/issues/85

        TODO: try to project a 3D position onto the 2D screen plane
        so can place text using the 2d only add_text  
        """
        vmtx = self.pl.camera.GetModelViewTransformMatrix()
        mtx = pv.trans_from_matrix(mtx)
        return mtx 

    def show(self):
        self.cp = self.pl.show()
        return self.cp


if __name__ == '__main__':

    name = sys.argv[1] if len(sys.argv) > 1 else "RoundaboutXY" 
    f = Flight.Load(name)

    pl = PVPlot3D() 

    pl.gaze_up_right_arrows( f.e3, f.g3, f.u3, f.r3, mag=0.5 )
    pl.add_grid()
    cp = pl.show()
pass

