#!/usr/bin/env python 
"""
CSGFoundry_CreateFromSimTest.py
==================================

"""

import numpy as np, os
from opticks.CSG.CSGFoundry import CSGFoundry


if __name__ == '__main__':

    np.set_printoptions(edgeitems=10) 
    a = CSGFoundry.Load("$FOLD", symbol="a")
    print(a.brief())




