#!/usr/bin/env python
import os, numpy as np
import pyvista as pv

if __name__ == '__main__':
    a = np.load(os.environ["NPY_PATH"])
    print(a)
    nrm = np.sum(a*a, axis=1)  

    


