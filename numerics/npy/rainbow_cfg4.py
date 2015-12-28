#!/usr/bin/env python
"""
# ChiSquared or KS
# http://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm 
# https://en.wikipedia.org/wiki/Propagation_of_uncertainty
# http://stats.stackexchange.com/questions/7400/how-to-assess-the-similarity-of-two-histograms
# http://www.hep.caltech.edu/~fcp/statistics/hypothesisTest/PoissonConsistency/PoissonConsistency.pdf
"""

import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

from env.numerics.npy.evt import Evt, costheta_, cross_
from env.numerics.npy.history import History
from env.numerics.npy.geometry import Boundary   
from env.numerics.npy.droplet import Droplet
from env.numerics.npy.fresnel import fresnel_factor

X,Y,Z,W = 0,1,2,3


deg = np.pi/180.
n2ref = 1.33257


def scatter_plot_cf(ax, a_evt, b_evt, axis=X, log_=False):
    db = np.arange(0,360,1)
    cnt = {}
    bns = {}
    ptc = {}
    j = -1
    for i,evt in enumerate([a_evt, b_evt]):
        dv = evt.a_deviation_angle(axis=axis)/deg
        ax.set_xlim(0,360)
        if len(dv) > 0:
            cnt[i], bns[i], ptc[i] = ax.hist(dv, bins=db,  log=log_, histtype='step', label=evt.label)
            j = i 
    pass
    if len(bns) == 2:
        assert np.all( bns[0] == bns[1] )

    return cnt, bns[j]


def cf_plot(evt_a, evt_b, label="", log_=False, ylim=[1,1e5]):

    fig = plt.figure()
    fig.suptitle("Rainbow cfg4 " + label )

    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])

    ax = fig.add_subplot(gs[0])

    c, bns = scatter_plot_cf(ax, evt_a, evt_b, axis=X, log_=log_)
    droplet.bow_angle_rectangles()
    ax.set_ylim(ylim)
    ax.legend()

    xlim = ax.get_xlim()

    ax = fig.add_subplot(gs[1])

    if len(c) == 2:
        a,b = c[0],c[1]

        msk = a+b > 0  
        c2 = np.zeros_like(a)
        c2[msk] = np.power(a-b,2)[msk]/(a+b)[msk]
        c2p = c2.sum()/len(a[msk])
        
        plt.plot( bns[:-1], c2, drawstyle='steps', label="chi2/ndf %4.2f" % c2p )
        ax.set_xlim(xlim) 
        ax.legend()

        ax.set_ylim([0,10]) 

        droplet.bow_angle_rectangles()



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(precision=4, linewidth=200)

    boundary = Boundary("Vacuum///MainH2OHale")
    droplet = Droplet(boundary)

    plt.ion()
    plt.close()

    label = "S"
    tag = "5"
    src = "torch"
    det = "rainbow"
    seqs = Droplet.seqhis([0,1,2,3,4,5,6,7],src="TO")
    not_ = False

    his_a = History.for_evt(tag="%s" % tag, src=src, det=det)
    his_b = History.for_evt(tag="-%s" % tag, src=src, det=det)

    sa = set(his_a.table.labels)
    sb = set(his_b.table.labels)

    sc = sorted(list(sa & sb), key=lambda _:his_a.table.label2count.get(_, None)) 

    ba = sb - sa
    print "Opticks but not G4, his_a.table(sa-sb)\n", his_a.table(sa - sb)
    print "G4 but not Opticks, his_b.table(sb-sa)\n", his_b.table(sb - sa)

    g4o = ["TO BT BR BR BR BR BR BR BR NA", "TO BT SC BR BR BR BR BR BR NA"]   # two largest G4 only, 482 23
    opo = ["TO BT BR BR BR BR BR BR BR BR", "TO BT BR BR BR BR BR BR BR BT" ]  # two largest Op only, 304 183 
    sq = g4o + opo

if 1:
    for seq in sq:

        seqs = [seq]
        evt_a =  Evt(tag=tag, src=src, det=det, label="%s Op" % label, seqs=seqs, not_=not_)
        evt_b =  Evt(tag="-%s" % tag, src=src, det=det, label="%s G4" % label, seqs=seqs, not_=not_)

        #sli = slice(0,15)
        sli = slice(None)
        evt_a.history_table(sli)
        evt_b.history_table(sli)

        cf_plot(evt_a, evt_b, label=seq, log_=False, ylim=None)



