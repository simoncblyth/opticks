#!/usr/bin/env python
"""
tgltf.py : Shakedown analytic geometry
==========================================================

Loads test events from Opticks

Create the events by running tgltf-transitional

Huh, top of cyl-z should not be there::

    In [8]: lpos[lpos[:,2] > 1500 ][:100]
    Out[8]: 
    A()sliced
    A([[ -367.125 ,   236.7812,  1535.    ,     1.    ],
           [  337.    , -1032.    ,  1535.    ,     1.    ],
           [  568.8125, -1328.9688,  1535.    ,     1.    ],
           [ 1212.875 ,  -858.375 ,  1535.    ,     1.    ],
           [  137.0625,  -371.6875,  1535.    ,     1.    ],
           [  849.6875,   997.6562,  1545.9814,     1.    ],
           [ -936.5625,   868.7812,  1547.71  ,     1.    ],
           [  196.3125,   411.9688,  1535.    ,     1.    ],
           [  -55.625 ,  -304.75  ,  1535.    ,     1.    ],
           [ -144.5   ,  -538.3125,  1535.    ,     1.    ],
           [ 1299.0625,  -612.9375,  1535.    ,     1.    ],
           [ -407.5   ,    13.3438,  1535.    ,     1.    ],
           [  865.375 ,   370.4062,  1535.    ,     1.    ],
           [  416.75  ,   478.5938,  1535.    ,     1.    ],
           [  431.75  ,   800.6875,  1535.    ,     1.    ],
           [   -8.5625,  1549.9375,  1526.9644,     1.    ],
           [  948.25  ,  -512.3438,  1535.    ,     1.    ],
           [  229.    ,   -32.5625,  1535.    ,     1.    ],
           [-1007.125 ,  -461.25  ,  1535.    ,     1.    ],
           [  -74.6875,  -607.125 ,  1535.    ,     1.    ],
           [  503.625 ,  -807.9062,  1535.    ,     1.    ],
           [  160.125 , -1057.0625,  1535.    ,     1.    ],
           [ -798.3125,    67.3125,  1535.    ,     1.    ],
           [-1278.25  ,   865.4062,  1535.    ,     1.    ],
           [ -509.625 ,   477.1562,  1535.    ,     1.    ],
           [ -141.875 ,  1289.5   ,  1535.    ,     1.    ],




"""
import os, sys, logging, argparse, numpy as np
import numpy.linalg as la

log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.nbase import vnorm
from opticks.ana.evt  import Evt

from opticks.analytic.sc  import gdml2gltf_main


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)

    os.environ['OPTICKS_QUERY']="range:3159:3160" 

    args = opticks_main(doc=__doc__, tag="1", src="torch", det="gltf" )

    sc = gdml2gltf_main(args)
    tx = sc.get_transform(3159)
    print tx

    itx = la.inv(tx)
    print itx


    log.info("tag %s src %s det %s  " % (args.utag,args.src,args.det))


    seqs=[]

    try:
        a = Evt(tag="%s" % args.utag, src=args.src, det=args.det, seqs=seqs, args=args)
    except IOError as err:
        log.fatal(err)
        #sys.exit(args.mrc)  this causes a sysrap-t test fail from lack of a tmp file
        sys.exit(0)


    log.info( " a : %s " % a.brief)

    print a.seqhis_ana.table

    a.sel = "TO SA"
    ox = a.ox


    print ox.shape   # masked array with those photons

    pos = ox[:,0,:4]
    pos[:,3] = 1.

    lpos = np.dot( pos, itx )





