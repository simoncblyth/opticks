#!/usr/bin/env python
"""
tdefault_distrib.py 
=============================================


"""
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = 18,10.2   # plt.gcf().get_size_inches()   after maximize
    import matplotlib.gridspec as gridspec
except ImportError:
    print "matplotlib missing : you need this to make plots"
    plt = None 

from opticks.ana.base import opticks_main
from opticks.ana.nbase import vnorm
from opticks.ana.evt  import Evt
from opticks.ana.cf import CF 
from opticks.ana.cfplot import cfplot, qwns_plot, qwn_plot, multiplot


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)
    args = opticks_main(tag="1", src="torch", det="concentric")
    log.info(" args %s " % repr(args))
    log.info("tag %s src %s det %s c2max %s  " % (args.utag,args.src,args.det, args.c2max))

    plt.ion()
    plt.close()

    seqs = ["49ccccd"]

    try:
        cf = CF(tag=args.tag, src=args.src, det=args.det, select=None, seqs=seqs )
    except IOError as err:
        log.fatal(err)
        sys.exit(args.mrc)

    cf.dump()

    irec = 6

    #multiplot(cf, pages=["XYZT","ABCR"])
  
    #qwn_plot( cf.ss[0], "T", -1, c2_ymax=2000)
    #qwn_plot( cf, "R", irec)
    #qwns_plot( cf, "XYZT", irec)
    #qwns_plot( cf, "ABCR", irec)


    binsx,ax,bx,lx = cf.rqwn("X",irec)
    binsy,ay,by,ly = cf.rqwn("Y",irec)
    binsz,az,bz,lz = cf.rqwn("Z",irec)

    axyz = np.array([[0,0,0],[2995.0267,0,0],[3004.9551,0,0],[3995.0491,0,0],[4004.9776,0,0],[4995.0716,0,0]])
    bxyz = np.array([[0,0,0],[2995.0267,0,0],[3004.9551,0,0],[3995.0491,0,0],[4004.9776,0,0],[4995.0716,0,0]])
   






 
