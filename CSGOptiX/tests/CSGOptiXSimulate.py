#!/usr/bin/env python
"""
::

    cx ; ipython -i tests/CSGOptiXSimulate.py


"""
import os, numpy as np

class CSGOptiXSimulate(object):
    FOLD = os.path.expandvars("/tmp/$USER/opticks/CSGOptiX/CSGOptiXSimulate")
    def __init__(self):
        p = np.load(os.path.join(self.FOLD, "photons.npy"))
        globals()["p"] = p 


if __name__ == '__main__':
    cxs = CSGOptiXSimulate()




