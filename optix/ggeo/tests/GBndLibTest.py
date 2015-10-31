#!/usr/bin/env python
"""
Compare old GBoundaryLib buffer with new GBndLib dynamically created oned

To get indices agreement in the optical buffer had to::

  cp GSurfaceIndexLocal.json   ~/.opticks/GSurfaceLib/order.json 
  cp GMaterialIndexLocal.json  ~/.opticks/GMaterialLib/order.json 

  ggv -G       # recreate geocache
  ggv --bnd    # recreate bnd buffers
  ggv --pybnd  # run this test

Why is order changing on each run ?  Probably not, presumably 
this was due to the move from digest based identity to shortname based identity. 

"""
import os
import numpy as np

load_ = lambda _:np.load(os.path.expandvars("$IDPATH/%s" % _)) 

def test_boundary_buffer():
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

def test_optical_buffer():
    # huh indices in optical.npy 1-based ? Yep npy- Index::add sets local indices like this
    w = load_("optical.npy").reshape(-1,4,4)  
    b = load_("GBndLib/GBndLibOptical.npy")

    assert w.shape == b.shape , ( w.shape, b.shape )
    for i in range(len(w)):
        if not np.all( w[i] == b[i] ):
            print 
            print i
            print w[i]
            print b[i] 
    pass
    assert np.all( w == b ) == True 



if __name__ == '__main__':
    test_boundary_buffer()
    test_optical_buffer()


