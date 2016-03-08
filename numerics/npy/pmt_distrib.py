#!/usr/bin/env python
"""
PmtInBox Opticks vs cfg4 distributions
==========================================

Without and with cfg4 runs::

   ggv-;ggv-pmt-test 
   ggv-;ggv-pmt-test --cfg4 

Visualize the cfg4 created evt in interop mode viewer::

   ggv-;ggv-pmt-test --cfg4 --load

Use G4 ui for G4 viz::

   ggv-;ggv-pmt-test --cfg4 --g4ui

Issues
-------

See pmt_test.py for the history of getting flags and materials into agreement.


"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from env.numerics.npy.evt import Evt
from env.numerics.npy.nbase import chi2

X,Y,Z,W = 0,1,2,3


class CF(object):
    def __init__(self, tag="4", src="torch", det="PmtInBox", seqs=[], subselect=None ):

        self.tag = tag
        self.src = src
        self.det = det
        self.seqs = seqs

        suptitle = "%s %s %s " % (det, src, tag )
        a = Evt(tag="%s" % tag, src=src, det=det, seqs=seqs)
        b = Evt(tag="-%s" % tag , src=src, det=det, seqs=seqs)

        log.info("CF a %s b %s " % (a.label, b.label )) 

        his = a.history.table.compare(b.history.table)
        mat = a.material.table.compare(b.material.table)

        self.suptitle = suptitle
        self.a = a
        self.b = b 
        self.his = his
        self.mat = mat

        self.ss = []
        if subselect is not None:
            self.init_subselect(subselect)

    def init_subselect(self, sli):
        """
        Spawn CF for each of the subselections, according to 
        slices of the history sequences.
        """
        for label in self.his.labels[sli]:
            seqs = [label]
            ss = self.spawn(seqs)
            self.ss.append(ss) 

    def __repr__(self):
        return "CF(%s,%s,%s,%s) " % (self.tag, self.src, self.det, repr(self.seqs))

    def spawn(self, seqs):
        return CF(self.tag, self.src, self.det, seqs)


    def dump_ranges(self, i):
        log.info("%s : dump_ranges %s " % (repr(self), i) )

        a = self.a
        b = self.b

        ap = a.rpost_(i)
        ar = np.linalg.norm(ap[:,:2],2,1)
        if len(ar)>0:
            print " ".join(map(lambda _:"%6.3f" % _, (ar.min(),ar.max())))

        bp = b.rpost_(0)
        br = np.linalg.norm(bp[:,:2],2,1)
        if len(br)>0:
            print " ".join(map(lambda _:"%6.3f" % _, (br.min(),br.max())))

    def dump_histories(self):
        print self.his
        #print self.mat

    def dump(self):
        self.dump_ranges(0)
        self.dump_histories()


def cf_plot(ax, aval, bval,  bins, label, log_=False):
    cnt = {}
    bns = {}
    ptc = {}
    cnt[0], bns[0], ptc[0] = ax.hist(aval, bins=bins,  log=log_, histtype='step', label=label)
    cnt[1], bns[1], ptc[1] = ax.hist(bval, bins=bins,  log=log_, histtype='step', label=label)
    return cnt, bns


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(precision=4, linewidth=200)

    plt.ion()
    plt.close()

    tag = "4"

    subselect = None
    subselect = slice(0,3)

    cf = CF(tag=tag, src="torch", det="PmtInBox", subselect=subselect )
    cf.dump()


    aval = cf.ss[0].a.rpost_(1)[:,3]
    bval = cf.ss[0].b.rpost_(1)[:,3]

    # how to choose bins that avoid compression artifacts 
    # timerange probably 0:100

    bins = np.linspace(min(aval.min(),bval.min()),max(aval.max(),bval.max()),128)

    fig = plt.figure()

    fig.suptitle(cf.suptitle)

    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])

    ax = fig.add_subplot(gs[0])

    ylim = [0,70000]
    ylim2 = [0, 10]
    label = "rec1 time"
    log_ = True
    c, bns = cf_plot(ax, aval, bval, bins=bins, label=label, log_=log_)
    assert len(c) == 2

    ax.set_ylim(ylim)
    ax.legend()

    xlim = ax.get_xlim()
    ax = fig.add_subplot(gs[1])

    a,b = c[0],c[1]

    c2, c2n = chi2(a, b, cut=30)
    c2p = c2.sum()/c2n
       
    label = "chi2/ndf %4.2f" % c2p
 
    plt.plot( bins[:-1], c2, drawstyle='steps', label=label )

    ax.set_xlim(xlim) 
    ax.legend()

    ax.set_ylim(ylim2) 




