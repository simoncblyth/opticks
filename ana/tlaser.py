#!/usr/bin/env python
"""
tlaser.py 
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

    args = opticks_main(doc=__doc__, tag="1", src="torch", det="laser", c2max=2.0, tagoffset=0)

    log.info("tag %s src %s det %s c2max %s  " % (args.utag,args.src,args.det, args.c2max))


    #a_seqs = ["8ccccd"]
    #b_seqs = ["8ccccccd"]
    a_seqs = ["8ccccd"]
    b_seqs = ["8ccccd"]

    try:
        a = Evt(tag="%s" % args.utag, src=args.src, det=args.det, seqs=a_seqs)
        b = Evt(tag="-%s" % args.utag , src=args.src, det=args.det, seqs=b_seqs)
    except IOError as err:
        log.fatal(err)
        sys.exit(args.mrc)
  

    log.info( " a : %s " % a.brief)
    log.info( " b : %s " % b.brief )

    print "A",a
    print "B",b



if 0:
    if a.valid:
        a0 = a.rpost_(0)
        #a0r = np.linalg.norm(a0[:,:2],2,1)
        a0r = vnorm(a0[:,:2])
        if len(a0r)>0:
            print " ".join(map(lambda _:"%6.3f" % _, (a0r.min(),a0r.max())))

    if b.valid:
        b0 = b.rpost_(0)
        #b0r = np.linalg.norm(b0[:,:2],2,1)
        b0r = vnorm(b0[:,:2])
        if len(b0r)>0:
            print " ".join(map(lambda _:"%6.3f" % _, (b0r.min(),b0r.max())))

if 1:
    Evt.compare_table(a,b, "seqhis_ana seqmat_ana".split(), lmx=20, c2max=None, cf=False)
    Evt.compare_table(a,b, "pflags_ana hflags_ana".split(), lmx=20, c2max=None, cf=False)

    #a.history_table(slice(0,20))
    #b.history_table(slice(0,20))


