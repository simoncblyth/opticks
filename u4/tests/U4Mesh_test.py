#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold
import pyvista as pv
SIZE = np.array([1280, 720])

def PolyData_FromTRI(f):
    """
    tri only : have to fiddle to get into format needed by PolyData 
    
    Can avoid that using TPD
    """
    tri = np.zeros( (len(f.tri), 4 ), dtype=np.int32 )
    tri[:,0] = 3 
    tri[:,1:] = f.tri    
    pd = pv.PolyData(f.vtx, tri)
    return pd 

def PolyData_FromFPD(f):
    """
    Note this may include quads as well as tri 
    """
    pd = pv.PolyData(f.vtx, f.fpd)
    return pd 

def PolyData_FromTPD(f):
    """
    tri only 
    """
    pd = pv.PolyData(f.vtx, f.tpd)
    return pd 




if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    #pd = PolyData_FromTRI(f)
    #label = "PolyData_FromTRI" 

    #pd = PolyData_FromTPD(f)
    #label = "PolyData_FromTPD" 

    pd = PolyData_FromFPD(f)
    label = "PolyData_FromFPD" 


    pl = pv.Plotter(window_size=SIZE*2)
    pl.add_text("%s" % label, position="upper_left")

    pl.add_mesh(pd, opacity=1.0, show_edges=True, lighting=True )
    pl.show()
pass


