#!/usr/bin/env python

import numpy as np

from opticks.CSG.CSGFoundry import CSGFoundry

cf = CSGFoundry.Load()  # sensitive to CFBASE_LOCAL and other such envvars

if __name__ == '__main__':
    print(cf)
    print(cf.sim)





