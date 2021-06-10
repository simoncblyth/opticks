#!/usr/bin/env python
"""


Repeat input photons and offset them 
using the polarization direction.::


          r = 5 
          j0 = -r//2  
          j1 =  r//2 


                       ^  direction 
           |  |  |  |  |
           |  |  |  |  |
           |  |  |  |  |
           |  |  |  |  |
           |  |  |  |  |
       -------------------------------> polarization 


    photon item (4,4)

      0: pos_x, pos_y, pos_z, time
      1: dir_x, dir_y, dir_z, weight  
      2: pol_x, pol_y, pol_z, wavelength
      4: flags 




"""
import numpy as np
from opticks.ana.input_photons import InputPhotons
import matplotlib.pyplot as plt
import pyvista as pv   


def mock_photons(o):
    p = np.zeros([o,4,4], dtype=np.float32)
    for k in range(o): 
        p[k,2,:3] = [0,0,1]
    pass
    return p 


def mplt(pp):

    pos_x, pos_y, pos_z = pp[:,0,0], pp[:,0,1], pp[:,0,2]
    dir_x, dir_y, dir_z = pp[:,1,0], pp[:,1,1], pp[:,1,2]

    plt.ion()
    fig = plt.figure(figsize=(6,5.5))
    ax = fig.add_subplot(projection='3d') 
    ax.scatter( pos_x, pos_y, pos_z )
    ax.scatter( pos_x+dir_x, pos_y+dir_y, pos_z+dir_z )
    fig.show()

def pvplt(pp):
    pl = pv.Plotter()

    pos = pp[:,0,:3]
    dir = pp[:,1,:3]
    pol = pp[:,2,:3]
    oth = np.cross( pol, dir )   
    mag = 1 
    pl.add_arrows( pos, dir, mag=mag*1.0,  color='#FF0000', point_size=2.0 )
    pl.add_arrows( pos, pol, mag=mag*1.0,  color='#00FF00', point_size=2.0 )
    pl.add_arrows( pos, oth, mag=mag*1.0,  color='#0000FF', point_size=2.0 )

    pl.show_grid()
    cpos = pl.show()


def test_parallelise_1d(p):
    r = 10 
    pp = InputPhotons.Parallelize1D(p, r, offset=True)
    pvplt(pp)
    return pp

def test_parallelise_2d(p):
    rr = [10,10] 
    pp = InputPhotons.Parallelize2D(p, rr, offset=True)
    pvplt(pp)
    return pp


if __name__ == '__main__':
    p = InputPhotons.GenerateCubeCorners()
    #p = InputPhotons.GenerateAxes()
    #p = p[0:1]

    pp = test_parallelise_2d(p)

