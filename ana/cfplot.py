#!/usr/bin/env python
"""
cfplot.py : Comparison Plotter with Chi2 Underplot 
======================================================


"""
import os, logging, numpy as np
from collections import OrderedDict as odict
log = logging.getLogger(__name__)


try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    print "matplotlib missing : you need this to make plots"
    plt = None


from opticks.ana.nbase import chi2

def _cf_plot(ax, aval, bval,  bins, labels,  log_=False):
    cnt = {}
    bns = {}
    ptc = {}
    cnt[0], bns[0], ptc[0] = ax.hist(aval, bins=bins,  log=log_, histtype='step', label=labels[0])
    cnt[1], bns[1], ptc[1] = ax.hist(bval, bins=bins,  log=log_, histtype='step', label=labels[1])
    return cnt, bns


def _chi2_plot(ax, bins, counts, cut=30):
    a,b = counts[0],counts[1]

    c2, c2n, c2c = chi2(a, b, cut=cut)
    c2p = c2.sum()/c2n
       
    label = "chi2/ndf %4.2f [%d]" % (c2p, c2c)

    ax.plot( bins[:-1], c2, drawstyle='steps', label=label )

    return c2p



def cfplot(fig, gss, bins, aval, bval, labels=["A","B"], log_=False, c2_cut=30, c2_ymax=10, logyfac=3., linyfac=1.3): 

    ax = fig.add_subplot(gss[0])

    counts_dict, bns = _cf_plot(ax, aval, bval, bins=bins, labels=labels, log_=log_)

    ymin = 1 if log_ else 0 
    ymax = max(map(lambda _:_.max(), counts_dict.values()))*(logyfac if log_ else linyfac)
    ylim = [ymin,ymax]

    ax.set_ylim(ylim)
    ax.legend()
    xlim = ax.get_xlim()


    ax = fig.add_subplot(gss[1])

    c2p = _chi2_plot(ax, bins, counts_dict, cut=c2_cut)  

    ax.set_xlim(xlim) 
    ax.legend()
    ax.set_ylim([0,c2_ymax]) 

    return c2p  





def qwns_plot(scf, qwns, irec, log_=False, c2_cut=30):

    fig = plt.figure()

    if irec < 0:
         irec += scf.nrec() 

    fig.suptitle(scf.suptitle(irec))

    nx = len(qwns)

    gs = gridspec.GridSpec(2, nx, height_ratios=[3,1])

    c2ps = []
    for ix in range(nx):

        gss = [gs[ix], gs[nx+ix]]

        qwn = qwns[ix]

        bins, aval, bval, labels = scf.rqwn(qwn, irec)

        log.info("%s %s " % (qwn, repr(labels) ))

        c2p = cfplot(fig, gss, bins, aval, bval, labels=labels, log_=log_, c2_cut=c2_cut )

        c2ps.append(c2p) 
    pass

    qd = odict(zip(list(qwns),c2ps))
    return qd


def qwn_plot(scf, qwn, irec, log_=False, c2_cut=30, c2_ymax=10):

    if irec < 0:
         irec += scf.nrec() 

    bins, aval, bval, labels = scf.rqwn(qwn, irec)

    fig = plt.figure()
    fig.suptitle(scf.suptitle(irec))

    nx,ix = 1,0
    gs = gridspec.GridSpec(2, nx, height_ratios=[3,1])
    gss = [gs[ix], gs[nx+ix]]

    c2p = cfplot(fig, gss, bins, aval, bval, labels=labels, log_=log_, c2_cut=c2_cut, c2_ymax=c2_ymax)
    c2ps = [c2p]

    #print "c2p", c2p

    qd = odict(zip(list(qwn),c2ps))
    return qd




def multiplot(cf, pages=["XYZT","ABCR"]):

    qwns = "".join(pages)
    dtype = [("key","|S64")] + [(q,np.float32) for q in list(qwns)]

    log_ = False
    c2_cut = 0.

    stat = np.recarray((cf.totrec,), dtype=dtype)

    ival = 0 
    for scf in cf.ss:
        nrec = scf.nrec()
        for irec in range(nrec):
            key = scf.suptitle(irec)

            od = odict()
            od.update(key=key) 

            for page in pages:
                qd = qwns_plot( scf, page, irec, log_, c2_cut)
                od.update(qd)
            pass

            stat[ival] = tuple(od.values())
            ival += 1
        pass
    pass

    np.save(os.path.expandvars("$TMP/stat.npy"),stat)

    #rst = recarray_as_rst(stat)
    #print rst 







if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(precision=4, linewidth=200)

    plt.ion()
    plt.close()

    aval = np.random.standard_normal(8000)
    bval = np.random.standard_normal(8000)
    bins = np.linspace(-4,4,200)
    log_ = False

    fig = plt.figure()
    fig.suptitle("cfplot test")

    nx = 4
    gs = gridspec.GridSpec(2, nx, height_ratios=[3,1])
    for ix in range(nx):
        gss = [gs[ix], gs[nx+ix]]
        cfplot(fig, gss, bins, aval, bval, labels=["A test", "B test"], log_=log_ )



