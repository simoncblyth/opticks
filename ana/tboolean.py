#!/usr/bin/env python
"""
tboolean.py 
=============================================




::

    ab.sel = "[TO] BT BT BT BT SA" 
    hh = ab.hh



"""
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.ab   import AB

if __name__ == '__main__':
    ok = opticks_main(doc=__doc__, tag="1", src="torch", det="tboolean-torus", smry=False)  

    print "ok.smry %d " % ok.smry 
    log.info(ok.brief)

    ab = AB(ok)

    print ab
    print "ab.a.metadata", ab.a.metadata
    print "ab.a.metadata.csgmeta0", ab.a.metadata.csgmeta0

    print ab.rpost_dv
    print ab.rpol_dv
    print ab.ox_dv


    rc = ab.RC 

    if not ok.ipython:
        log.info("early exit as non-interactive")
        sys.exit(rc)
    else:
        pass
    pass

    a = ab.a
    b = ab.b
    #ab.aselhis = "TO BT BT SA"     # dev aligned comparisons
  
   

    

