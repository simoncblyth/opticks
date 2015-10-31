#!/usr/bin/env python
"""
Compare old GBoundaryLib buffer with new GBndLib dynamically created oned
"""
import os
import numpy as np

load_ = lambda _:np.load(os.path.expandvars("$IDPATH/%s" % _)) 

if __name__ == '__main__':

    w = load_("wavelength.npy").reshape(-1,4,39,4)

    b = load_("GBndLib/GBndLib.npy")

    assert w.shape == b.shape , ( w.shape, b.shape )

    num_quad = 4 

    for j in range(num_quad):
        print "quad %d " % j 
        for i in range(len(w)):
            assert np.all( w[i][j] == b[i][j] ) 
        pass

    assert np.all( w == b ) == True 

