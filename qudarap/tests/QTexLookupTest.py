#!/usr/bin/env python
"""
::

    ~/o/qudarap/tests/QTexLookupTest.sh
    ~/o/qudarap/tests/QTexLookupTest.sh pdb
    ~/o/qudarap/tests/QTexLookupTest.sh ana

"""
import os, sys, logging, numpy as np
from opticks.ana.fold import Fold

log = logging.getLogger(__name__)

if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO)

     f = Fold.Load(symbol="f")
     print(repr(f))

     print("f.origin\n", f.origin )
     print("f.lookup\n", f.lookup )

     match = np.all( f.lookup == f.origin )
     rc = 0 if match else 1
     print(f" match {match} rc {rc} \n")
     sys.exit( rc )
