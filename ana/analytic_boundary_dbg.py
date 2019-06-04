#!/usr/bin/env python
"""
tboolean.py: CSG boolean geometry tracetest 
=======================================================

For analysis of photon_buffer written by oxrap/cu/generate.cu:tracetest
which is intended for testing geometrical intersection only, with 
no propagation.

Unsigned boundaries 1 and 124 all from CSG intersection, the 215 was a
abitrary marker written by evaluative_csg::

    In [43]: flg[:100]
    Out[43]: 
    A()sliced
    A([[  0,   0,   1, 215],
           [  0,   0, 123,   0],
           [  0,   0,   1, 215],
           [  0,   0, 123,   0],
           [  0,   0, 123,   0],
           [  0,   0,   1, 215],
           [  0,   0, 123,   0],
           [  0,   0,   1, 215],
           [  0,   0, 123,   0],
           [  0,   0, 123,   0],
           [  0,   0, 123,   0],
           [  0,   0,   1, 215],
           [  0,   0, 123,   0],
           [  0,   0, 124, 215],
           [  0,   0,   1, 215],
           [  0,   0,   1, 215],
           [  0,   0,   1, 215],
           [  0,   0,   1, 215],
           [  0,   0,   1, 215],
           [  0,   0,   1, 215],


    In [46]: np.unique(flg[flg[:,3] == 215][:,2])
    Out[46]: 
    A()sliced
    A([  1, 124], dtype=uint32)


    In [48]: count_unique_sorted(flg[:,2])   # unsigned 0-based boundaries 
    Out[48]: 
    array([[    1, 58822],
           [  123, 35537],
           [  124,  5641]], dtype=uint64)


    In [57]: t0[np.where(flg[:,2] == 1)]
    A([ 149.8889,  149.8889,  349.8889, ...,  348.1284,  349.8889,  349.8889], dtype=float32)

    In [58]: t0[np.where(flg[:,2] == 124)]
    A([ 51.6622,  47.394 ,  61.4086, ...,  61.0178,  66.7235,  47.1538], dtype=float32)

    In [59]: t0[np.where(flg[:,2] == 123)]
    A([ 1400.    ,  1400.    ,  1400.    , ...,  2021.8895,  1400.    ,  1400.    ], dtype=float32)


::

    In [13]: count_unique(ib)   # signed 1-based boundaries encode inner/outer
    Out[13]: 
    array([[  -125.,   4696.],
           [    -2.,  58316.],
           [     2.,    506.],
           [   124.,  35537.],
           [   125.,    945.]])

    In [35]: count_unique(ub)   # unsigned 0-based boundaries, for lookup against the blib,   58316 + 506 = 58822,  4696 + 945 = 5641
    Out[35]: 
    array([[    1, 58822],
           [  123, 35537],
           [  124,  5641]], dtype=uint64)



    In [33]: print "\n".join(["%3d : %s " % (i, n) for i,n in enumerate(blib.names)])
      0 : Vacuum///Vacuum 
      1 : Vacuum///Rock 
      2 : Rock///Air 
      3 : Air/NearPoolCoverSurface//PPE 
      4 : Air///Aluminium 
      5 : Aluminium///Foam 
      6 : Foam///Bakelite 
    ...
    119 : OwsWater/NearOutInPiperSurface//PVC 
    120 : OwsWater/NearOutOutPiperSurface//PVC 
    121 : DeadWater/LegInDeadTubSurface//ADTableStainlessSteel 
    122 : Rock///RadRock 


"""

import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt 
#from opticks.ana.nbase import count_unique_sorted  # doenst work with signed
from opticks.ana.nbase import count_unique
from opticks.ana.evt import Evt
from opticks.ana.proplib import PropLib

X,Y,Z,W = 0,1,2,3


if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    #args = opticks_main(tag="1", det="boolean", src="torch")
    ok = opticks_main()


    blib = PropLib("GBndLib")

    evt = Evt(tag=ok.tag, det=ok.det, src=ok.src, pfx=ok.pfx, args=ok)

    if not evt.valid:
       log.fatal("failed to load evt %s " % repr(args))
       sys.exit(1) 

    ox = evt.ox

    ## assume layout written by oxrap/cu/generate.cu:tracetest ##

    p0 = ox[:,0,:W]
    d0 = ox[:,1,:W]
    t0 = ox[:,1,W]
    flg = ox[:,3].view(np.uint32)  # u-flags 

    p1 = p0 + np.repeat(t0,3).reshape(-1,3)*d0   # intersect position


    ub = flg[:,2]                   # unsigned boundary 
    ib = ox[:,2,W].view(np.int32)   # signed boundary 

    b = 1 
    #b = 123
    #b = 124 


    thin = slice(0,None,100)
    s = np.where(ub == b)[0]

    plt.ion()
    plt.close()


    #plt.scatter( p0[s,X], p0[s,Y] ) 
    plt.scatter( p1[s,X], p1[s,Y] )


    plt.show()

