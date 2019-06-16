#!/usr/bin/env python
"""

::

   ipython $(which GPtsTest.py) -i -- 0


"""
import os, sys 
import numpy as np

if __name__ == '__main__':

    imm = sys.argv[1] if len(sys.argv) > 0 else 0
    path = os.path.expandvars("$TMP/GGeo/GPtsTest/%(imm)s" % locals())
    name = "idxBuffer.npy"
    print(path)

    apath = os.path.join(path, "parts", name)
    bpath = os.path.join(path, "parts2", name)

    a = np.load(apath)
    b = np.load(bpath)

    au = len(np.unique(a[:,0]))
    bu = len(np.unique(b[:,0]))
   
    print("A:%s %s au:%d" % (a.shape,apath,au) )
    print("B:%s %s bu:%d" % (b.shape,bpath,bu) )
     
    print(a[:10])
    print(b[:10])

    assert np.all( a[:,1] == b[:,1])
    assert np.all( a[:,2] == b[:,2])
    assert np.all( a[:,3] == b[:,3])


