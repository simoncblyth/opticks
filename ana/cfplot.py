#!/usr/bin/env python
"""
cfplot.py : Comparison Plotter with Chi2 Underplot 
======================================================

To control this warning, see the rcParam `figure.max_num_figures



"""
import os, logging, numpy as np
from collections import OrderedDict as odict
from opticks.ana.base import opticks_main
from opticks.ana.cfh import CFH 
log = logging.getLogger(__name__)


try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    plt.rcParams["figure.max_open_warning"] = 200    # default is 20
except ImportError:
    print "matplotlib missing : you need this to make plots"
    plt = None



def cfplot(fig, gss, h): 

    ax = fig.add_subplot(gss[0])

    ax.plot( h.bins[:-1], h.ahis , drawstyle="steps", label=h.la  )
    ax.plot( h.bins[:-1], h.bhis , drawstyle="steps", label=h.lb  )

    if h.log:
        ax.set_yscale('log')

    ax.set_ylim(h.ylim)
    ax.legend()

    xlim = ax.get_xlim()

    ax = fig.add_subplot(gss[1])

    ax.plot( h.bins[:-1], h.chi2, drawstyle='steps', label=h.c2label )

    ax.set_xlim(xlim) 
    ax.legend()
    ax.set_ylim([0,h.c2_ymax]) 


def qwns_plot( ok, hh, suptitle ):
    nhh = len(hh)
    nxm = 4 
    if nhh > nxm:
        # pagination 
        for p in range(nhh/nxm):
            phh = hh[p*nxm:(p+1)*nxm]
            qwns_plot( ok, phh, suptitle + " (%d)" % p )   
        pass
    else:
        fig = plt.figure(figsize=ok.figsize)
        fig.suptitle(suptitle)
        ny = 2 
        nx = len(hh)
        gs = gridspec.GridSpec(ny, nx, height_ratios=[3,1])
        for ix in range(nx):
            gss = [gs[ix], gs[nx+ix]]
            h = hh[ix]
            cfplot(fig, gss, hh[ix] )
        pass

def one_cfplot(ok, h):
    fig = plt.figure(figsize=ok.figsize)
    fig.suptitle(h.suptitle)
    ny = 2
    nx = 1
    gs = gridspec.GridSpec(ny, nx, height_ratios=[3,1])
    for ix in range(nx):
        gss = [gs[ix], gs[nx+ix]]
        cfplot(fig, gss, h )
    pass


def multiplot(ok, ab, start=0, stop=5, log_=False):
    """
    """
    pages = ok.qwn.split(",")

    for i,isel in enumerate(range(start, stop)):

        ab.sel = slice(isel, isel+1)
        nrec = ab.nrec

        for irec in range(nrec):

            ab.irec = irec 
            suptitle = ab.suptitle

            log.info("multiplot irec %d nrec %d suptitle %s " % (irec, nrec, suptitle))

            for page in pages:
                hh = ab.rhist(page, irec, log_ )
                qwns_plot( hh, suptitle )
            pass
        pass
    pass







if __name__ == '__main__':
    ok = opticks_main()
    print ok

    plt.ion()
    plt.close()

    from opticks.ana.ab import AB

    h = AB.rrandhist()

    one_cfplot(ok, h) 



