#!/usr/bin/env python
"""
pv_polyData_test.py 
======================

::

 
    OPT=0 ./pv_polyData_test.sh  # split line 
    OPT=1 ./pv_polyData_test.sh  # contiguous line 


OPT 0 and 1 look exactly the same when using 
the default line presentation. 

When using STY=tube the split or contiguous 
distinction is visible. 

"""

import os, numpy as np
OPT = int(os.environ.get("OPT","0"))
STY = os.environ.get("STY","")
SIZE = np.array([1280, 720] )
import pyvista as pv


if __name__ == '__main__':

     points = np.array( [[0,0,0], [1,0,0], [1,1,0], [1,1,1]], dtype=np.float32 ) 
     labels = np.array(["000","100","110","111"]) 

     ## line connectivity array
     if OPT == 0:     
         lines = np.hstack([[2, 0, 1], 
                            [2, 1, 2],
                            [2, 2, 3]])  
     elif OPT == 1: 
         lines = np.array( [4,0,1,2,3], dtype=np.int32 ) 
     else:
         pass
     pass

     print("OPT:%d lines:%s STY:%s " % (OPT, str(lines), STY ))

     pl = pv.Plotter(window_size=SIZE*2 )
     #pl.show_grid()
     pd = pv.PolyData() 
     pd.points = points
     pd.lines = lines
     pd["points_label"] = labels

     if STY == "tube":
         tube = pd.tube(radius=0.1) 
         pl.add_mesh(tube)
     else:
         pl.add_mesh(pd)
     pass
     pl.add_point_labels(pd, "points_label", point_size=20, font_size=36)


     pl.show()

    
