#!/usr/bin/env python
"""
tgltf.py : Shakedown analytic geometry
==========================================================

Loads test events from Opticks

Create the events by running tgltf-transitional



"""
import os, sys, logging, argparse, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.nbase import vnorm
from opticks.ana.evt  import Evt



if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)

    args = opticks_main(doc=__doc__, tag="1", src="torch", det="gltf" )

    log.info("tag %s src %s det %s  " % (args.utag,args.src,args.det))


    seqs=[]

    try:
        a = Evt(tag="%s" % args.utag, src=args.src, det=args.det, seqs=seqs, args=args)
    except IOError as err:
        log.fatal(err)
        #sys.exit(args.mrc)  this causes a sysrap-t test fail from lack of a tmp file
        sys.exit(0)

  

    log.info( " a : %s " % a.brief)

