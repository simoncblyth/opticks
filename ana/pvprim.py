#!/usr/bin/env python 
"""
pvprim.py
=============

https://docs.pyvista.org/examples/00-load/create-geometric-objects.html
"""

import numpy as np
import pyvista as pv

_white = "ffffff"
_red = "ff0000"
_green = "00ff00"
_blue = "0000ff"

DTYPE = np.float64
SIZE = np.array([1280, 720])


"""
Rotation about Y transforms X axis to Z axis::

     
       Z                      X
       |  Y                   |  Y
       | /                    | /
       |/                     |/
       +----- X        Z -----+


"""



x2z = np.array( 
              [[  0., 0., -1., 0],
               [  0., 1.,  0., 0],
               [  1., 0.,  0., 0],
               [  0., 0.,  0., 0]] )


if __name__ == '__main__':


    pl = pv.Plotter(window_size=SIZE*2, shape=(3,3) )

    cyl = pv.Cylinder(direction=(0,0,1))
    arrow = pv.Arrow(direction=(0,0,1))
    cone = pv.Cone(direction=(0,0,1))

    sphere = pv.Sphere()
    plane = pv.Plane()
    line = pv.Line()
    box = pv.Box()
    poly = pv.Polygon()
    disc = pv.Disc()

    pl.subplot(0, 0)
    pl.add_mesh(cyl, color=_red, show_edges=True)
    pl.show_grid()

    pl.subplot(0, 1)
    pl.add_mesh(arrow, color=_green, show_edges=True)
    pl.show_grid()

    pl.subplot(0, 2)
    pl.add_mesh(sphere, color=_blue, show_edges=True)
    pl.show_grid()

    # Middle row
    pl.subplot(1, 0)
    pl.add_mesh(plane, color="tan", show_edges=True)
    pl.show_grid()

    pl.subplot(1, 1)
    pl.add_mesh(line, color="tan", line_width=3)
    pl.show_grid()

    pl.subplot(1, 2)
    pl.add_mesh(box, color="tan", show_edges=True)
    pl.show_grid()

    # Bottom row
    pl.subplot(2, 0)
    pl.add_mesh(cone, color="tan", show_edges=True)
    pl.show_grid()

    pl.subplot(2, 1)
    pl.add_mesh(poly, color="tan", show_edges=True)
    pl.show_grid()

    pl.subplot(2, 2)
    pl.add_mesh(disc, color="tan", show_edges=True)
    pl.show_grid()

    cp = pl.show()
    print(cp)



