#!/usr/bin/env python
"""
tpmt_skimmer.py: Following positions of PMT skimmers
======================================================

Creates plot showing step by step average positions of 
all photons with a specific history, namely: "TO BT BR BR BT SA"
and tabulates the min/max/mid positions.

Expected Output
-----------------

.. code-block:: py

    In [1]: run tpmt_skimmer.py
    WARNING:opticks.ana.evt:init_index PmtInBox/torch/-5 : TO BT BR BR BT SA finds too few (ps)phosel uniques : 1
    WARNING:opticks.ana.evt:init_index PmtInBox/torch/-5 : TO BT BR BR BT SA finds too few (rs)recsel uniques : 1
    WARNING:opticks.ana.evt:init_index PmtInBox/torch/-5 : TO BT BR BR BT SA finds too few (rsr)reshaped-recsel uniques : 1
    A(Op) PmtInBox/torch/5 : TO BT BR BR BT SA 
      0 z:    300.000    300.000    300.000   r:     98.999     98.999     98.999   t:      0.098      0.098      0.098   smry m1/m2   4/ 14 MO/Py  -28 ( 27)  13:TO  
      1 z:     67.559     67.559     67.559   r:     98.999     98.999     98.999   t:      1.251      1.251      1.251   smry m1/m2  14/  4 Py/MO   28 ( 27)  12:BT  
      2 z:     50.832     50.832     50.832   r:    100.372    100.372    100.372   t:      1.331      1.331      1.331   smry m1/m2  14/ 11 Py/OV -125 (124)  11:BR  
      3 z:     35.551     35.551     35.551   r:     93.176     93.176     93.176   t:      1.416      1.416      1.416   smry m1/m2  14/  4 Py/MO   28 ( 27)  11:BR  
      4 z:     19.181     19.181     19.181   r:     89.001     89.001     89.001   t:      1.495      1.495      1.495   smry m1/m2   4/ 12 MO/Rk  124 (123)  12:BT  
      5 z:   -300.000   -300.000   -300.000   r:     26.569     26.569     26.569   t:      3.107      3.107      3.107   smry m1/m2   4/ 12 MO/Rk  124 (123)   8:SA  
    B(G4) PmtInBox/torch/-5 : TO BT BR BR BT SA 
      0 z:    300.000    300.000    300.000   r:     98.999     98.999     98.999   t:      0.098      0.098      0.098   smry m1/m2   4/  0 MO/?0?    0 ( -1)  13:TO  
      1 z:     67.559     67.559     67.559   r:     98.999     98.999     98.999   t:      1.251      1.251      1.251   smry m1/m2  14/  0 Py/?0?    0 ( -1)  12:BT  
      2 z:     50.832     50.832     50.832   r:    100.372    100.372    100.372   t:      1.331      1.331      1.331   smry m1/m2  14/  0 Py/?0?    0 ( -1)  11:BR  
      3 z:     35.551     35.551     35.551   r:     93.176     93.176     93.176   t:      1.416      1.416      1.416   smry m1/m2  14/  0 Py/?0?    0 ( -1)  11:BR  
      4 z:     19.181     19.181     19.181   r:     89.001     89.001     89.001   t:      1.495      1.495      1.495   smry m1/m2   4/  0 MO/?0?    0 ( -1)  12:BT  
      5 z:   -300.000   -300.000   -300.000   r:     26.569     26.569     26.569   t:      3.107      3.107      3.107   smry m1/m2   4/  0 MO/?0?    0 ( -1)   8:SA  


See Also
------------

:doc:`pmt_skimmer_debug`
       Debugging Opticks TIR with pmt_skimmer.py 


"""

import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt

from opticks.ana.base import opticks_main
from opticks.ana.evt import Evt
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
    args = opticks_main(tag="10", src="torch", det="PmtInBox")
    np.set_printoptions(precision=4, linewidth=200)

    plt.ion()
    plt.close()

    fig = plt.figure()

    pmt = Pmt()
    ALL, PYREX, VACUUM, CATHODE, BOTTOM, DYNODE = None,0,1,2,3,4
    pts = pmt.parts(ALL)
    one_plot(fig, pmt, pts, axes=ZX, clip=True)

    pol = False
  
    # aseqs=["TO BT BR BT BT BT BT BT BT SA"]   before fixing the TIR bug this was what was happening
    aseqs=["TO BT BR BR BT SA"]
    bseqs=["TO BT BR BR BT SA"]
    na = len(aseqs[0].split())
    nb = len(bseqs[0].split())

    try:
        a = Evt(tag="%s" % args.tag, src=args.src, det=args.det, seqs=aseqs)
        b = Evt(tag="-%s" % args.tag, src=args.src, det=args.det, seqs=bseqs)
    except IOError as err:
        log.fatal(err)
        sys.exit(args.mrc)

    if a.valid:
        log.info("A : plot zrt_profile %s " % a.label)
        print "A(Op) %s " % a.label
        a_zrt = a.zrt_profile(na, pol=pol)
        zr_plot(a_zrt)
    else:
        log.warning("failed to load A")

    if b.valid:
        log.info("B : plot zrt_profile %s " % b.label)
        print "B(G4) %s " % b.label
        b_zrt = b.zrt_profile(nb, pol=pol)
        zr_plot(b_zrt, neg=True)
    else:
        log.warning("failed to load B")


    


