#!/usr/bin/env python

import os, logging, numpy as np
from opticks.CSG.scan import Scan 

log = logging.getLogger(__name__)

try:
    import pyvista as pv
except ImportError:
    pv = None
pass

if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO)
     geom = os.environ.get("GEOM", "iphi")  
     base = os.path.expandvars("$TMP/CSGNodeScanTest")
     scan = Scan.Load( os.path.join(base, geom) )
     Scan.Plot(scan)



