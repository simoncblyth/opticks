#!/usr/bin/env python
"""
See also ana/geocache.bash 
"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

idp_ = lambda _:os.path.expandvars("$IDPATH/%s" % _ )
idp2_ = lambda _:os.path.expandvars("$IDPATH2/%s" % _ )



def cflib(aa, bb):
    """
    Compare buffers between two geocache
    """
    assert aa.shape == bb.shape
    print aa.shape

    for i in range(len(aa)):
        a = aa[i]  
        b = bb[i]  
        assert len(a) == 2 
        assert len(b) == 2 

        g0 = a[0] - b[0] 
        g1 = a[1] - b[1] 

        assert g0.shape == g1.shape

        print i, g0.shape, "g0max: ", np.max(g0), "g1max: ", np.max(g1)
    pass


def cflib_test():
    #rel = "GMaterialLib/GMaterialLib.npy"
    rel = "GSurfaceLib/GSurfaceLib.npy"

    aa = np.load(idp_(rel))
    bb = np.load(idp2_(rel))
    cflib(aa,bb)
 


if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO)

    assert 0, "ancient code depending in IDPATH"

