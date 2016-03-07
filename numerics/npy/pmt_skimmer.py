#!/usr/bin/env python
"""

Issues
--------

[FIXED] Timing Off
~~~~~~~~~~~~~~~~~~~~

Refractive index of MO different ? Nope, checked that.

G4 impinges magical GROUPVEL that changes light velocity.  Regained match via CPropLib::m_groupvel_kludge
to set he groupvel property to the phase velocity.

::

    A(Op)
      0 z    300.000    300.000    300.000 r     98.999     98.999     98.999  t      0.098      0.098      0.098    smry m1/m2   4/ 14 MO/Py  -28 ( 27)  13:TO  
      1 z     67.559     67.559     67.559 r     98.999     98.999     98.999  t      1.251      1.251      1.251    smry m1/m2  14/  4 Py/MO   28 ( 27)  12:BT  
      2 z     50.832     50.832     50.832 r    100.372    100.372    100.372  t      1.331      1.331      1.331    smry m1/m2  14/ 11 Py/OV -125 (124)  11:BR  
      3 z     35.551     35.551     35.551 r     93.176     93.176     93.176  t      1.416      1.416      1.416    smry m1/m2  14/  4 Py/MO   28 ( 27)  12:BT  
      4 z      2.005      2.005      2.005 r     81.137     81.137     81.137  t      1.532      1.532      1.532    smry m1/m2   4/ 14 MO/Py  -28 ( 27)  12:BT  
      5 z   -114.115   -114.115   -114.115 r     42.253     42.253     42.253  t      1.953      1.953      1.953    smry m1/m2  14/ 13 Py/Vm  -29 ( 28)  12:BT  
      6 z   -123.875   -123.875   -123.875 r     39.250     39.250     39.250  t      1.990      1.990      1.990    smry m1/m2  14/ 13 Py/Vm  -29 ( 28)  12:BT  
      7 z   -150.810   -150.810   -150.810 r     39.250     39.250     39.250  t      2.051      2.051      2.051    smry m1/m2   4/ 14 MO/Py  -28 ( 27)  12:BT  
      8 z   -169.002   -169.002   -169.002 r     39.250     39.250     39.250  t      2.081      2.081      2.081    smry m1/m2   4/ 12 MO/Rk  124 (123)  12:BT  
      9 z   -300.000   -300.000   -300.000 r     39.250     39.250     39.250  t      2.301      2.301      2.301    smry m1/m2   4/ 12 MO/Rk  124 (123)   8:SA  
    B(G4)
      0 z    300.000    300.000    300.000 r     98.999     98.999     98.999  t      0.098      0.098      0.098    smry m1/m2   4/  0 MO/?0?    0 ( -1)  13:TO  
      1 z     67.559     67.559     67.559 r     98.999     98.999     98.999  t      1.325      1.325      1.325    smry m1/m2  14/  0 Py/?0?    0 ( -1)  12:BT  
      2 z     50.832     50.832     50.832 r    100.372    100.372    100.372  t      1.410      1.410      1.410    smry m1/m2  14/  0 Py/?0?    0 ( -1)  11:BR  
      3 z     35.551     35.551     35.551 r     93.176     93.176     93.176  t      1.489      1.489      1.489    smry m1/m2  14/  0 Py/?0?    0 ( -1)  11:BR  
      4 z     19.181     19.181     19.181 r     89.001     89.001     89.001  t      1.575      1.575      1.575    smry m1/m2   4/  0 MO/?0?    0 ( -1)  12:BT  
      5 z   -300.000   -300.000   -300.000 r     26.569     26.569     26.569  t      3.290      3.290      3.290    smry m1/m2   4/  0 MO/?0?    0 ( -1)   8:SA  



[FIXED] getting BT when looks like need BR at very tangential reflection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fixed Opticks TIR bug::

    optixrap-/propagate.h


    241 __device__ void propagate_at_boundary_geant4_style( Photon& p, State& s, curandState &rng)
    242 {
    243     // see g4op-/G4OpBoundaryProcess.cc annotations to follow this
    ...
    291     const float TransCoeff =  tir ? 0.0f : n2c2*E2_total_t/n1c1 ;
    292     //  above 0.0f was until 2016/3/4 incorrectly a 1.0f 
    293     //  resulting in TIR yielding BT where BR is expected
    294     
    295     bool reflect = curand_uniform(&rng) > TransCoeff  ;
    296     
    297     p.direction = reflect
    298                     ?  
    299                        p.direction + 2.0f*c1*s.surface_normal
    300                     :  
    301                        eta*p.direction + (eta_c1 - c2)*s.surface_normal
    302                     ;


After fix  (with the GROUPVEL phase velocity kludge in CPropLib) and fixpol override::

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


::

     ggv --bnd

     ( 27) om:               MineralOil os:                          is:                          im:                    Pyrex
     ( 28) om:                    Pyrex os:                          is:                          im:                   Vacuum
     ...
     (122) om:                     Rock os:                          is:                          im:                  RadRock




"""

import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from env.numerics.npy.evt import Evt
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

    tag = "5"
    pol = False
  
    # aseqs=["TO BT BR BT BT BT BT BT BT SA"]   before fixing the TIR bug this was what was happening
    aseqs=["TO BT BR BR BT SA"]
    na = len(aseqs[0].split())
    a = Evt(tag="%s" % tag, src="torch", det="PmtInBox", seqs=aseqs)

    bseqs=["TO BT BR BR BT SA"]
    nb = len(bseqs[0].split())
    b = Evt(tag="-%s" % tag, src="torch", det="PmtInBox", seqs=bseqs)

    print "A(Op) %s " % a.label
    a_zrt = a.zrt_profile(na, pol=pol)
    zr_plot(a_zrt)

    print "B(G4) %s " % b.label
    b_zrt = b.zrt_profile(nb, pol=pol)
    zr_plot(b_zrt, neg=True)





