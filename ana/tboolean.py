#!/usr/bin/env python
"""
tboolean.py 
=============================================

"""
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.ab   import AB

if __name__ == '__main__':
    ok = opticks_main(doc=__doc__, tag="1", src="torch", det="tboolean-torus", smry=True)  

    print "ok.smry %d " % ok.smry 
    log.info(ok.brief)

    ab = AB(ok)
    print ab
    print ab.a.metadata
    print ab.a.metadata.csgmeta0

    if not ok.ipython:
        log.info("early exit as non-interactive")
        sys.exit(0)

    #ab.sel = "[TO] BT BT BT BT SA" 
    #
    #hh = ab.hh

    

    

