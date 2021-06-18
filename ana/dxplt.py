#!/usr/bin/env python
"""
dxplt.py : pyvista 3d plotting G4 dx.npy photon step points within selections
=============================================================================== 

::

   run dxplt.py 



* https://docs.pyvista.org/examples/00-load/create-spline.html

"""
import numpy as np
from opticks.ana.ab import AB
import pyvista as pv


def lines_from_points(points):
    """
    See ~/env/python/pv/create-spline.py 

    https://docs.pyvista.org/examples/00-load/create-spline.html

    Given an array of points, make a line set

    array([[2, 0, 1],
           [2, 1, 2],
           [2, 2, 3],
           [2, 3, 4],
           [2, 4, 5],
           [2, 5, 6]])

    """
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly



if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    ok = opticks_main(pfx="tds3ip", src="natural")
    ab = AB(ok)  
    a = ab.a    # dx only filled for G4:b 
    b = ab.b 

    #a_sel = "TO SC BT SR BT SA"     # 0x8cac6d
    #b_sel = "TO SC BT SR BT BT SA"  # 0x8ccac6d

    sel = "TO SC BT BT AB"   # scatters that get back into the LS from the Water
    n = len(sel.split())   

    a.sel = sel 
    b.sel = sel

    #pos = b.dx[:,:n,0,:3]      ## deluxe double buffer is G4 only 
    #pos = b.rpost()[:,:,:3]    ## rpost from rx buffer has same info with domain compression   
    pos = a.rpost()[:,:,:3]      

    pl = pv.Plotter()

    for i in range(len(pos)):
        line = lines_from_points(pos[i])
        line["scalars"] = np.arange(line.n_points)
        tube = line.tube(radius=3)
        pl.add_mesh(tube)
    pass
    pl.show()

