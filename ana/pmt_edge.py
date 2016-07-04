#!/usr/bin/env python
"""
"""

import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from opticks.ana.evt import Evt
from env.nuwa.detdesc.pmt.plot import Pmt, one_plot


X,Y,Z,W = 0,1,2,3

ZX = [Z,X]
ZY = [Z,Y]
XY = [X,Y]

ZMIN,ZMAX,ZAVG,RMIN,RMAX,RAVG,TMIN,TMAX,TAVG = 0,1,2,3,4,5,6,7,8


def zr_plot(data, neg=False):
    xx = data[:,ZAVG]
    yy = data[:,RAVG]
    if neg:yy=-yy

    labels = map(str, range(len(data)))
    plt.plot(xx, yy, "o")
    for label, x, y in zip(labels, xx,yy):
        plt.annotate(label, xy = (x, y), xytext = (-3,10), textcoords = 'offset points')



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(precision=4, linewidth=200)

    plt.ion()
    plt.close()

    fig = plt.figure()

    pmt = Pmt()
    ALL, PYREX, VACUUM, CATHODE, BOTTOM, DYNODE = None,0,1,2,3,4
    pts = pmt.parts(ALL)
    one_plot(fig, pmt, pts, axes=ZX, clip=True)

    tag = "4"
    pol = False
  

    #seqs = ["TO BT BT SA"]
    seqs = ["TO BT SA"]

    ns = len(seqs[0].split())
    a = Evt(tag="%s" % tag, src="torch", det="PmtInBox", seqs=seqs)
    b = Evt(tag="-%s" % tag, src="torch", det="PmtInBox", seqs=seqs)

    a0 = a.rpost_(0)
    a0r = np.linalg.norm(a0[:,:2],2,1)
    b0 = b.rpost_(0)
    b0r = np.linalg.norm(b0[:,:2],2,1)

    plt.hist(a0r, bins=100)
    plt.hist(b0r, bins=100)


    print "A(Op) %s " % a.label
    a_zrt = a.zrt_profile(ns, pol=pol)
    zr_plot(a_zrt)

    print "B(G4) %s " % b.label
    b_zrt = b.zrt_profile(ns, pol=pol)
    zr_plot(b_zrt, neg=True)





