#!/usr/bin/env python
import os, numpy as np
import pyvista as pv
from opticks.ana.pvplt import * 

if __name__ == '__main__':
    a = np.load(os.environ["NPY_PATH"])
    print(a)
    nrm = np.sum(a*a, axis=1)  

    pl = pvplt_plotter(label="G4RandomToolsTest.py")
    pl.add_points(a[:,:3])
    cp = pl.show()




    


