#!/usr/bin/env python
"""
pmt_edge.py : PMT edge skimmers debugging
=============================================

Tabulates min, max and mid positions of photon steps
for specific step sequences such as:  TO BT BT SA 
standing for TORCH, BOUNDARY_TRANSMIT*2, SURFACE_ABSORB


::

    A(Op) PmtInBox/torch/4 : TO BT BT SA 
      0 z:    300.000    300.000    300.000   r:     97.086     99.999     98.543   t:      0.100      0.100      0.100   smry m1/m2   4/ 14 MO/Py  -28 ( 27)  13:TO  
      1 z:     63.100     74.270     68.685   r:     97.086     99.999     98.543   t:      1.216      1.272      1.244   smry m1/m2  14/  4 Py/MO   28 ( 27)  12:BT  
      2 z:     43.828     54.787     49.307   r:     98.843    101.008     99.925   t:      1.312      1.365      1.339   smry m1/m2   4/ 12 MO/Rk  124 (123)  12:BT  
      3 z:   -300.000   -300.000   -300.000   r:    135.556    164.406    149.981   t:      3.075      3.095      3.085   smry m1/m2   4/ 12 MO/Rk  124 (123)   8:SA  
    B(G4) PmtInBox/torch/-4 : TO BT BT SA 
      0 z:    300.000    300.000    300.000   r:     97.086    100.002     98.544   t:      0.100      0.100      0.100   smry m1/m2   4/  0 MO/?0?    0 ( -1)  13:TO  
      1 z:     63.100     74.270     68.685   r:     97.086    100.002     98.544   t:      1.291      1.350      1.320   smry m1/m2  14/  0 Py/?0?    0 ( -1)  12:BT  
      2 z:     43.828     54.787     49.307   r:     98.844    101.006     99.925   t:      1.391      1.439      1.415   smry m1/m2   4/  0 MO/?0?    0 ( -1)  12:BT  
      3 z:   -300.000   -300.000   -300.000   r:    135.554    164.395    149.974   t:      3.264      3.292      3.278   smry m1/m2   4/  0 MO/?0?    0 ( -1)   8:SA  


"""

import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt

from opticks.ana.base import opticks_environment
from opticks.ana.evt import Evt

#from env.nuwa.detdesc.pmt.plot import Pmt, one_plot
from opticks.ana.pmt.plot import Pmt, one_plot

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
    opticks_environment()


    plt.ion()
    plt.close()

    fig = plt.figure()

    pmt = Pmt()
    ALL, PYREX, VACUUM, CATHODE, BOTTOM, DYNODE = None,0,1,2,3,4
    pts = pmt.parts(ALL)
    one_plot(fig, pmt, pts, axes=ZX, clip=True)

    tag = "4"
    pol = False
  

    seqs = ["TO BT BT SA"]
    #seqs = ["TO BT SA"]

    ns = len(seqs[0].split())
    a = Evt(tag="%s" % tag, src="torch", det="PmtInBox", seqs=seqs)
    b = Evt(tag="-%s" % tag, src="torch", det="PmtInBox", seqs=seqs)

    a0 = a.rpost_(0)
    a0r = np.linalg.norm(a0[:,:2],2,1)
    b0 = b.rpost_(0)
    b0r = np.linalg.norm(b0[:,:2],2,1)

    #plt.hist(a0r, bins=100)
    #plt.hist(b0r, bins=100)


    print "A(Op) %s " % a.label
    a_zrt = a.zrt_profile(ns, pol=pol)
    zr_plot(a_zrt)

    print "B(G4) %s " % b.label
    b_zrt = b.zrt_profile(ns, pol=pol)
    zr_plot(b_zrt, neg=True)





