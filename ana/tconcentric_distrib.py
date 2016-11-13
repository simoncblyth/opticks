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
from opticks.ana.cfplot import cfplot, qwns_plot, multiplot


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


    start, stop, qwns = 0,5, "XYZT,ABCR" 

    multiplot(ab, start, stop, qwns )

    st = ab.stats( start, stop, qwns )


 
