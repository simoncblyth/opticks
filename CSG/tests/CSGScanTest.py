#!/usr/bin/env python
"""

::

   GEOMETRY=zsph ipython -i CSGScanTest.py 



"""
import os
import numpy as np
from glob import glob

try:
    import matplotlib.pyplot as plt 
except ImportError:
    plt = None
pass

try:
    import pyvista as pv
except ImportError:
    pv = None
pass

def plot3d(pos, grid=False):
    """
    https://docs.pyvista.org/plotting/plotting.html

    q:close
    v:isometric camera
    +/-: increase/decrease point size
    """
    pl = pv.Plotter()
    pl.add_points(pos, color='#FFFFFF', point_size=2.0 )  
    if grid:
        pl.show_grid()
    pass
    cp = pl.show()
    return cp


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


class CSGScanTest(object):
    def __init__(self, path):
        a = np.load(path) 

        ori = a[:,0]  ; valid_isect = a[:,0,3].view(np.int32)  
        dir = a[:,1]
        post = a[:,2]
        isect = a[:,3]

        tot = len(a)
        hit = np.count_nonzero( valid_isect == 1 )  
        miss = np.count_nonzero( valid_isect == 0 )  


        self.path = path 
        self.a = a

        self.ori = ori    ; self.valid_isect = valid_isect
        self.dir = dir
        self.post = post
        self.isect = isect

        self.tot = tot 
        self.hit = hit 
        self.miss = miss

    def __repr__(self):
        return "%s : tot %d hit %d miss %d " % (self.path, self.tot, self.hit, self.miss)
 



def plot2d(st):
    plt.ion()
    fig, axs = plt.subplots(1)
    if not type(axs) is np.ndarray: axs = [axs] 

    ax = axs[0]
    ax.set_aspect('equal')
    ax.scatter( st.post[:,0], st.post[:,2], s=0.1 )
    ax.scatter( st.ori[:,0],  st.ori[:,2] )
    scale = 10.
    ax.scatter( st.ori[:,0] + st.dir[:,0]*scale, st.ori[:,2]+st.dir[:,2]*scale )
    fig.show()



if __name__ == '__main__':


    solid = os.environ.get("CSGSCANTEST_SOLID", "elli")

    #solid = "sphe"
    #solid = "zsph"
    #solid = "cone"
    #solid = "vcub" 
    #solid = "vtet" 
    #solid = "hype"
    #solid = "box3"
    #solid = "plan"
    #solid = "slab"
    #solid = "cyli"
    #solid = "disc"
    #solid = "elli"
    #solid = "ubsp"
    #solid = "ibsp"
    #solid = "dbsp"

    print("CSGSCANTESTSOLID : %s " % solid )

    #scan = "circle"
    scan = "rectangle"

    base = os.environ.get("CSGSCANTEST_BASE", os.path.expandvars("/tmp/$USER/opticks/CSGScanTest"))
    path = "%(base)s/%(scan)s_scan/%(solid)s.npy" % locals()

    st = CSGScanTest(path)
    print(st)

    #plot2d( st );
    plot3d( st.post[:,:3] )   # intersect position 
    #plot3d_arrows( st.post[:,:3], st.isect[:,:3], mag=10 )  # intersect position and normal direction arrows


