#!/usr/bin/env python
"""
cfplot.py : Comparison Plotter with Chi2 Underplot 
======================================================

To control this warning, see the rcParam `figure.max_num_figures



"""
import os, logging, numpy as np
from collections import OrderedDict as odict
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



def one_cfplot(h):
    fig = plt.figure()

    fig.suptitle(h.suptitle)

    ny = 2
    nx = 1

    gs = gridspec.GridSpec(ny, nx, height_ratios=[3,1])
    for ix in range(nx):
        gss = [gs[ix], gs[nx+ix]]
        cfplot(fig, gss, h )
    pass




def qwns_plot(ab, qwns, irec, log_=False ):

    log.info("qwns_plot(scf, \"%s\", %d, log_=%s  )" % (qwns, irec, log_ )) 

    fig = plt.figure()

    ab.irec = irec

    fig.suptitle(ab.suptitle)

    ny = 2 
    nx = len(qwns)

    gs = gridspec.GridSpec(ny, nx, height_ratios=[3,1])

    c2ps = []
    for ix in range(nx):

        gss = [gs[ix], gs[nx+ix]]

        qwn = qwns[ix]

        h = ab.rhist(qwn, irec)

        h.log = log_

        cfplot(fig, gss, h )

        c2ps.append(h.c2p) 
    pass

    qd = odict(zip(list(qwns),c2ps))
    return qd



def qwn_plot(ab, qwn, irec, log_=False ):

    ab.irec = irec      

    h = ab.rhist(qwn, irec)

    h.log = log_

    fig = plt.figure()
    fig.suptitle(ab.suptitle)

    nx,ix = 1,0
    gs = gridspec.GridSpec(2, nx, height_ratios=[3,1])
    gss = [gs[ix], gs[nx+ix]]

    cfplot(fig, gss, h )
    
    c2ps = [h.c2p]

    #print "c2p", c2p

    qd = odict(zip(list(qwn),c2ps))
    return qd



def multiplot(ab, pages=["XYZT","ABCR"], sli=slice(0,5)):
    """
    Inflexible approach taken for recording distrib chi2 
    is making this inflexible to use
    """
    qwns = "".join(pages)
    dtype = [("key","|S64")] + [(q,np.float32) for q in list(qwns)]

    log_ = False

    trs = ab.totrec(sli.start, sli.stop)
    nrs = ab.nrecs(sli.start, sli.stop)

    log.info(" multiplot : trs %d nrs %s " % ( trs, repr(nrs)) )
        
    stat = np.recarray((trs,), dtype=dtype)
    ival = 0
 
    for i,isel in enumerate(range(sli.start, sli.stop)):

        ab.sel = slice(isel, isel+1)
        nr = ab.nrec
        assert nrs[i] == nr, (i, nrs[i], nr )  

        for irec in range(nr):

            ab.irec = irec 
            key = ab.suptitle
            log.info("multiplot irec %d nrec %d ival %d key %s " % (irec, nr, ival, key))

            od = odict()
            od.update(key=key) 

            for page in pages:
                qd = qwns_plot( ab, page, irec, log_ )
                od.update(qd)
            pass

            stat[ival] = tuple(od.values())
            ival += 1
        pass
    pass

    assert ival == trs, (ival, trs )

    np.save(os.path.expandvars("$TMP/stat.npy"),stat)  # make_rst_table.py reads this and dumps RST table
    return stat 

    # a = np.load(os.path.expandvars("$TMP/stat.npy"))
    #rst = recarray_as_rst(stat)
    #print rst 




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=4, linewidth=200)

    plt.ion()
    plt.close()

    from opticks.ana.ab import AB

    h = AB.rrandhist()

    #ctx = {'det':"concentric", 'tag':"1", 'qwn':"X", 'irec':"5", 'seq':"TO_BT_BT_BT_BT_DR_SA" }
    #cfp = CFP(ctx)
    #h = cfp()

    one_cfplot(h) 



