#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':
    t = Fold.Load(); 


    np.all( t.snap_isect[:,:,2,3] == 100. )  # huh: big tmin ?




