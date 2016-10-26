#!/usr/bin/env python
"""
tboxlaser.py 
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



if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)

    args = opticks_main(doc=__doc__, tag="1", src="torch", det="boxlaser", c2max=2.0, tagoffset=0)

    log.info("tag %s src %s det %s c2max %s  " % (args.utag,args.src,args.det, args.c2max))

    a_seqs = []
    b_seqs = []

    try:
        a = Evt(tag="%s" % args.utag, src=args.src, det=args.det, seqs=a_seqs)
        b = Evt(tag="-%s" % args.utag , src=args.src, det=args.det, seqs=b_seqs)
    except IOError as err:
        log.fatal(err)
        sys.exit(args.mrc)

    print "A",a
    print "B",b

    log.info( " a : %s " % a.brief)
    log.info( " b : %s " % b.brief )

    tables = ["seqhis_ana"] + ["seqhis_ana_%d" % imsk for imsk in range(1,8)] + ["seqmat_ana"] 
    Evt.compare_table(a,b, tables, lmx=20, c2max=None, cf=True)

    Evt.compare_table(a,b, "pflags_ana hflags_ana".split(), lmx=20, c2max=None, cf=True)



