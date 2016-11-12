#!/usr/bin/env python
"""
tconcentric_distrib.py 
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
from opticks.ana.ab import AB 
from opticks.ana.cfplot import cfplot, qwns_plot, qwn_plot, multiplot


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)
    ok = opticks_main(tag="1", src="torch", det="concentric")
    log.info(" ok %s " % repr(ok.brief))

    plt.ion()
    plt.close()

    try:
        ab = AB(ok)
    except IOError as err:
        log.fatal(err)
        sys.exit(ok.mrc)
    
    print ab


    st = multiplot(ab, pages=["XYZT","ABCR"], sli=slice(0,5))

    #ab.sel = slice(8,9)
    #qwns_plot(ab, "XYZT", 5, log_=False, c2_cut=0 )

    #st = multiplot(cf, pages=["XYZT"])
 
    #log_ = False
    #c2_cut = 0 
 
    #scf = cf.ss[0]
    #nrec = scf.nrec()
    #nrec = 1 
    #for irec in range(nrec):
    #    key = scf.suptitle(irec)
    #    page = "XYZT"
    #    qd = qwns_plot( scf, page, irec, log_, c2_cut)
    #    print "qd", qd
    #

    #qwns_plot( cf.ss[0], "XYZT", 0 ) 


    #irec = 6
    #qwn_plot( cf.ss[0], "T", -1, c2_ymax=2000)
    #qwn_plot( cf, "R", irec)
    #qwns_plot( cf, "XYZT", irec)
    #qwns_plot( cf, "ABCR", irec)

    #binsx,ax,bx,lx = cf.rqwn("X",irec)
    #binsy,ay,by,ly = cf.rqwn("Y",irec)
    #binsz,az,bz,lz = cf.rqwn("Z",irec)




 
