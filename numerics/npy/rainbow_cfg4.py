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


from env.numerics.npy.evt import Evt, History, costheta_, cross_
from env.numerics.npy.geometry import Boundary   
from env.numerics.npy.droplet import Droplet
from env.numerics.npy.fresnel import fresnel_factor

X,Y,Z,W = 0,1,2,3


deg = np.pi/180.
n2ref = 1.33257


def scatter_plot_cf(ax, a_evt, b_evt, axis=X):
    db = np.arange(0,360,1)
    cnt = {}
    bns = {}
    ptc = {}
    for i,evt in enumerate([a_evt, b_evt]):
        dv = evt.a_deviation_angle(axis=axis)/deg
        ax.set_xlim(0,360)
        ax.set_ylim(1,1e5)
        cnt[i], bns[i], ptc[i] = ax.hist(dv, bins=db,  log=True, histtype='step', label=evt.label)
    pass
    assert np.all( bns[0] == bns[1] )
    return cnt, bns[0]



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
    seqs = Droplet.seqhis([0,1,2,3,4,5,6,7])
    not_ = True

    his_b = History.for_evt(tag="-%s" % tag, src=src, det=det)

    evt_a =  Evt(tag=tag, src=src, det=det, label="%s Op" % label, seqs=seqs, not_=not_)
    evt_b =  Evt(tag="-%s" % tag, src=src, det=det, label="%s G4" % label, seqs=seqs, not_=not_)

    #sli = slice(0,15)
    sli = slice(None)
    evt_a.history_table(sli)
    evt_b.history_table(sli)

    fig = plt.figure()
    fig.suptitle("Rainbow cfg4")

    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])

    ax = fig.add_subplot(gs[0])

    c, bns = scatter_plot_cf(ax, evt_a, evt_b, axis=X)
    droplet.bow_angle_rectangles()
    ax.legend()

    xlim = ax.get_xlim()

    ax = fig.add_subplot(gs[1])

    a,b = c[0],c[1]

    msk = a+b > 0  

    c2 = np.zeros_like(a)
    c2[msk] = np.power(a-b,2)[msk]/(a+b)[msk]
    c2p = c2.sum()/len(a)

    
    plt.plot( bns[:-1], c2, drawstyle='steps', label="chi2/ndf %4.2f" % c2p )
    ax.set_xlim(xlim) 
    ax.legend()

    ax.set_ylim([0,10]) 

    droplet.bow_angle_rectangles()


