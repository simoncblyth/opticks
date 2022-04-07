#!/usr/bin/env python
"""

https://docs.pyvista.org/examples/00-load/create-spline.html

"""
import numpy as np
import pyvista as pv
from opticks.ana.pvplt import * 

def make_points():
    """
    In [1]: points
    Out[1]: 
    array([[ 0.   ,  5.   , -2.   ],
           [ 1.216,  4.685, -1.96 ],
           [ 2.277,  4.092, -1.919],
           [ 3.126,  3.278, -1.879],
           [ 3.722,  2.309, -1.838],
           [ 4.042,  1.257, -1.798],
           [ 4.084,  0.195, -1.758],
           [ 3.865, -0.809, -1.717],
           [ 3.415, -1.693, -1.677],

    """
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return np.column_stack((x, y, z))

def make_line_cells(num_points):
    """
    Cells of 2 indices like this yield visible divisions between the line segments

    In [2]: cells[:5]
    Out[2]: 
    array([[2, 0, 1],
           [2, 1, 2],
           [2, 2, 3],
           [2, 3, 4],
           [2, 4, 5]])

    In [3]: cells[-5:]
    Out[3]: 
    array([[ 2, 94, 95],
           [ 2, 95, 96],
           [ 2, 96, 97],
           [ 2, 97, 98],
           [ 2, 98, 99]])


    """
    dtype = np.int_
    shape = (num_points - 1, 3)
    cells = np.zeros( shape, dtype=dtype)
    cells[:, 0] = 2 
    cells[:, 1] = np.arange(0, num_points - 1, dtype=dtype )
    cells[:, 2] = cells[:,1] + 1 
    return cells  

def make_manual_cells():
    """
    disjoint spline
    """
    dtype = np.int_
    cells = np.zeros( (2,11), dtype=dtype )
    cells[0] = [10,  0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ] 
    cells[1] = [10, 20,21,22,23,24,25,26,27,28,29 ]   
    return cells

def make_single_cells(n):
    """
    All the indices in a single cell yields a tube without visible segments

    :param n: int
    :return: for n=5  np.array( [5, 0,1,2,3,4], dtype=np.int_ ) 
    """
    cells = np.zeros( n+1, dtype=np.int_ )
    cells[0] = n 
    cells[1:] = np.arange(n) 
    return cells 

def make_points_lines_polydata(points, lines):
    """
    :param points: float positions (n,3)
    :param lines: int 

    Given an array of points, make a line set
    """
    poly = pv.PolyData()
    poly.points = points
    poly.lines = lines
    return poly

if __name__ == '__main__':
    points = make_points()

    num_points = len(points) 

    #all_cells = make_line_cells(num_points)
    #cells = all_cells[:20] + all_cells[-20:]    ## segments 
    #cells = all_cells[:20] 

    cells = make_manual_cells()
  
    #cells = make_single_cells(num_points)

    line = make_points_lines_polydata(points, cells)
    line["scalars"] = np.arange(line.n_points)

    pl = pvplt_plotter()
    tube = line.tube(radius=0.1)

    pl.add_mesh( tube, smooth_shading=True)
    #pl.add_mesh( line, smooth_shading=True, line_width=5 )

    pl.show()



