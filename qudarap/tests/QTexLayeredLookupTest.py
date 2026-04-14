#!/usr/bin/env python
"""
::

    ~/o/qudarap/tests/QTexLayeredLookupTest.sh
    ~/o/qudarap/tests/QTexLayeredLookupTest.sh pdb
    ~/o/qudarap/tests/QTexLayeredLookupTest.sh ana

"""
import os, sys, logging, numpy as np
from opticks.ana.fold import Fold

log = logging.getLogger(__name__)

if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO)

     f = Fold.Load(symbol="f")
     print(repr(f))

     assert f.origin.shape == f.lookup.shape
     d,h,w,p = f.origin.shape
     assert p == 1

     for l in range(d):
         print(f"f.origin[{l}].reshape({h},{w})\n", f.origin[l].reshape(h,w) )
     pass
     for l in range(d):
         print(f"f.lookup[{l}].reshape({h},{w})\n", f.lookup[l].reshape(h,w) )
     pass

     match = np.all( f.lookup == f.origin )
     rc = 0 if match else 1
     print(f" match {match} rc {rc} \n")
     sys.exit( rc )
