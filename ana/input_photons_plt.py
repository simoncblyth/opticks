#!/usr/bin/env python
"""

"""
import os, sys, numpy as np
from opticks.ana.input_photons import InputPhotons

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

    def pos_dir_pol_arrows(self, pos, dir, pol, mag=0.5):
        self.pl.add_arrows( pos, dir, mag=mag*1.0,  color='#FF0000', point_size=2.0 ) 
        self.pl.add_arrows( pos, pol, mag=mag*0.5,  color='#00FF00', point_size=2.0 ) 

    def show(self):
        self.cp = self.pl.show()
        return self.cp


if __name__ == '__main__':
    ip = InputPhotons()
    print(ip)

    pl = PVPlot3D() 

    pos = ip.p[:1,0,:3]
    dir = ip.p[:1,1,:3]
    pol = ip.p[:1,2,:3] 

    pl.pos_dir_pol_arrows( pos, dir, pol, mag=0.5 )
    pl.add_grid()
    cp = pl.show()
pass

