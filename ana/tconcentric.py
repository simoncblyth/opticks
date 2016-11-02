#!/usr/bin/env python
"""
tconcentric.py 
=============================================

Loads test events from Opticks and Geant4 and 
created by OKG4Test and 
compares their bounce histories.

"""
import os, sys, logging, argparse, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.nbase import vnorm
from opticks.ana.evt  import Evt
from opticks.ana.cf   import CF


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)

    args = opticks_main(doc=__doc__, tag="1", src="torch", det="concentric", c2max=2.0, tagoffset=0,  dbgseqhis=0, lmx=20, prohis=False, promat=False, dbgzero=False, cmx=0)
    log.info("tag %s src %s det %s c2max %s  " % (args.utag,args.src,args.det, args.c2max))

    cf = CF(tag=args.tag, src=args.src, det=args.det, dbgseqhis=args.dbgseqhis, lmx=args.lmx, prohis=args.prohis, promat=args.promat, dbgzero=args.dbgzero, cmx=args.cmx)



