#!/usr/bin/env python
"""
"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.geocache import keydir
from opticks.ana.prim import Dir

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    ok = os.environ["OPTICKS_KEY"]
    kd = keydir(ok)

    pv = np.loadtxt(os.path.join(kd, "GNodeLib/PVNames.txt" ), dtype="|S100" )
    lv = np.loadtxt(os.path.join(kd, "GNodeLib/LVNames.txt" ), dtype="|S100" )
    print pv


 
