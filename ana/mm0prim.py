#!/usr/bin/env python
"""
Hmm need to make connection to the volume traversal index 
"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.geocache import keydir
from opticks.ana.prim import Dir

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    ok = os.environ["OPTICKS_KEY"]
    kd = keydir(ok)

    names = True
    if names:
        pvn = np.loadtxt(os.path.join(kd, "GNodeLib/PVNames.txt" ), dtype="|S100" )
        lvn = np.loadtxt(os.path.join(kd, "GNodeLib/LVNames.txt" ), dtype="|S100" )
    else:
        pvn = None
        lvn = None
    pass

    log.info(kd)
    assert os.path.exists(kd), kd 
    os.environ["IDPATH"] = kd 

    d = Dir(os.path.expandvars("$IDPATH/GParts/0"))
    #print d     
    pp = d.prims

    sli = slice(0,None)

    for p in pp[sli]:
        if p.lvIdx in [8,9]: continue   # too many  
        if p.numParts > 1: continue    # skip compounds for now

        vol = p.idx[0]
        pv = pvn[vol]
        lv = lvn[vol]
        print(repr(p)) 
        print(pv)
        print(lv)

        

    pass


