#!/usr/bin/env python 
"""
cylinder.py
=============


"""

import numpy as np
#import pyvista as pv
#pv.set_plot_theme("dark")
from opticks.ana.pvplt import * 
from opticks.ana.make_rotation_matrix import make_rotation_matrix

DTYPE = np.float64

X = np.array([1,0,0,0], dtype=DTYPE)
Y = np.array([0,1,0,0], dtype=DTYPE)
Z = np.array([0,0,1,0], dtype=DTYPE)
O = np.array([0,0,0,1], dtype=DTYPE)


def make_transforms_cube_corners():
    trs = np.zeros( (8,4,4) )
    for i in range(len(trs)):
        m = np.eye(4)
        tx = sc if i & 0x1 else -sc 
        ty = sc if i & 0x2 else -sc 
        tz = sc if i & 0x4 else -sc 
        m[3] = (tx,ty,tz,1)
        trs[i] = m.T
    pass
    return trs

def make_transforms_around_cylinder(radius, halfheight, num_ring=10, num_in_ring=16):
    zz = np.linspace( -halfheight, halfheight, num_ring ) 

    phi = np.linspace( 0, 2*np.pi, num_in_ring+1 )[:-1]
    xy_outwards = np.zeros( (len(phi), 4) )
    xy_outwards[:,0] = np.cos(phi)
    xy_outwards[:,1] = np.sin(phi)
    xy_outwards[:,2] = 0. 
    xy_outwards[:,3] = 0. 

    assert len(zz) == num_ring 
    assert len(xy_outwards) == num_in_ring 

    num_tr = num_ring*num_in_ring 
    trs = np.zeros( (num_tr, 4, 4) )
    for i in range(num_ring):
        z = zz[i]
        for j in range(num_in_ring):
            outwards = xy_outwards[j] 
            m4 = make_rotation_matrix(Z, outwards)   
            # HUH: expected -outwards to point the arrows inwards
            # presumably this is due to the transpose 
            # done for pyvista
            pos = outwards*radius 
            pos[2] = z 
            m4[3,:3] = pos[:3]   
            ## HMM: somewhat dirty stuffing the translation in like that 
            ## as its a bit unclear in what multiplication order it corresponds to 
            ## should really multiply matrices 
            idx = i*num_in_ring + j
            trs[idx] = m4.T       ## seems pyvista needs transposed 
        pass
    pass
    return trs 


if __name__ == '__main__':
 
    sc = 5 
    radius = sc
    halfheight = sc
    trs = make_transforms_around_cylinder(radius, halfheight)

    arrows = []
    for i in range(len(trs)):
        arr = pv.Arrow(direction=Z[:3])
        arr.transform(trs[i])
        arrows.append(arr)
    pass
    cyl = pv.Cylinder(center=O[:3], direction=Z[:3], radius=radius, height=2*halfheight) 

    pl = pvplt_plotter()
    pl.add_mesh(cyl, style="wireframe")
    for arr in arrows:
        pl.add_mesh(arr)
    pass
    pl.show()


