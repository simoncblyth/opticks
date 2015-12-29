#!/usr/bin/env python
"""
# ChiSquared or KS
# http://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm 
# https://en.wikipedia.org/wiki/Propagation_of_uncertainty
# http://stats.stackexchange.com/questions/7400/how-to-assess-the-similarity-of-two-histograms
# http://www.hep.caltech.edu/~fcp/statistics/hypothesisTest/PoissonConsistency/PoissonConsistency.pdf


TODO:

* try living without step-by-step recording, 
  to see the peformance impact of doing so 
  and of having ginormous record and sequence 
  arrays in the context 

  * need to get ox only plotting to work, 
    this requires primary recording to get the side

* revive compute only mode, ie without OpenGL involvement

"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

from env.numerics.npy.evt import Evt, costheta_, cross_
from env.numerics.npy.history import History, AbbFlags
from env.numerics.npy.geometry import Boundary   
from env.numerics.npy.droplet import Droplet
from env.numerics.npy.fresnel import fresnel_factor
from env.numerics.npy.nbase import chi2

X,Y,Z,W = 0,1,2,3

deg = np.pi/180.
n2ref = 1.33257


def a_scatter_plot_cf(ax, a_evt, b_evt, log_=False):
    db = np.arange(0,360,1)

    incident = np.array([0,0,-1])
    cnt = {}
    bns = {}
    ptc = {}
    j = -1
    for i,evt in enumerate([a_evt, b_evt]):
        dv = evt.a_deviation_angle(axis=X, incident=incident)/deg
        ax.set_xlim(0,360)
        if len(dv) > 0:
            cnt[i], bns[i], ptc[i] = ax.hist(dv, bins=db,  log=log_, histtype='step', label=evt.label)
            j = i 
    pass
    if len(bns) == 2:
        assert np.all( bns[0] == bns[1] )

    if j == -1:
        bns = None
    else:
        bns = bns[j] 

    return cnt, bns


def cf_plot(evt_a, evt_b, label="", log_=False, ylim=[1,1e5], ylim2=[0,10]):

    tim_a = " ".join(map(lambda f:"%5.2f" % f, map(float, filter(None, evt_a.tdii['propagate']) )))
    tim_b = " ".join(map(lambda f:"%5.2f" % f, map(float, filter(None, evt_b.tdii['propagate']) )))

    fig = plt.figure()
    fig.suptitle("Rainbow cfg4 " + label + "[" + tim_a + "] [" + tim_b + "]"  )

    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])

    ax = fig.add_subplot(gs[0])

    c, bns = a_scatter_plot_cf(ax, evt_a, evt_b, log_=log_)
    droplet.bow_angle_rectangles()
    ax.set_ylim(ylim)
    ax.legend()

    xlim = ax.get_xlim()

    ax = fig.add_subplot(gs[1])

    if len(c) == 2:
        a,b = c[0],c[1]

        c2, c2n = chi2(a, b, cut=30)
        c2p = c2.sum()/c2n
        
        plt.plot( bns[:-1], c2, drawstyle='steps', label="chi2/ndf %4.2f" % c2p )
        ax.set_xlim(xlim) 
        ax.legend()

        ax.set_ylim(ylim2) 

        droplet.bow_angle_rectangles()



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(precision=4, linewidth=200)

    boundary = Boundary("Vacuum///MainH2OHale")
    droplet = Droplet(boundary)
    af = AbbFlags()

    plt.ion()
    plt.close()

    rec = False
    tag = "5"
    src = "torch"
    det = "rainbow"
    log_ = True
    not_ = False

    if det == "rainbow":
       if tag == "5":
           label = "S-Pol"
       elif tag == "6":
           label = "P-Pol"
       else:
           label = "no label"


    if rec:
        seqs = Droplet.seqhis([0,1,2,3,4,5,6,7],src="TO")

        his_a = History.for_evt(tag="%s" % tag, src=src, det=det)
        his_b = History.for_evt(tag="-%s" % tag, src=src, det=det)

        cf = his_a.table.compare(his_b.table)
        print cf

        sa = set(his_a.table.labels)
        sb = set(his_b.table.labels)
        sc = sorted(list(sa & sb), key=lambda _:his_a.table.label2count.get(_, None)) 

        print "Opticks but not G4, his_a.table(sa-sb)\n", his_a.table(sa - sb)
        print "G4 but not Opticks, his_b.table(sb-sa)\n", his_b.table(sb - sa)
        sq = [None]

    else:
        sq = [None]

    for seq in sq:

        seqs = [] if seq is None else [seq]
        evt_a =  Evt(tag=tag, src=src, det=det, label="%s Op" % label, seqs=seqs, not_=not_, rec=rec)
        evt_b =  Evt(tag="-%s" % tag, src=src, det=det, label="%s G4" % label, seqs=seqs, not_=not_, rec=rec)

        #sli = slice(0,15)
        #sli = slice(None)
        #evt_a.history_table(sli)
        #evt_b.history_table(sli)


    if 1:
        cf_plot(evt_a, evt_b, label=str(seqs), log_=log_, ylim=[0.8,4e4],ylim2=None)



