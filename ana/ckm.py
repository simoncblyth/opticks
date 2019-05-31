#!/usr/bin/env python
"""
::

    In [14]: evt.rpost_(slice(0,5)).shape
    Out[14]: (500000, 5, 4)


"""
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.evt import Evt

if __name__ == '__main__':

    args = opticks_main(tag="1",src="natural", det="g4live", pfx="OKG4Test", doc=__doc__)
    np.set_printoptions(suppress=True, precision=3)

    print("pfx:%s" % args.pfx ) 

    a = Evt(tag=args.tag, src=args.src, det=args.det, pfx=args.pfx, seqs=[], args=args)
    print("a")
    print(a)

    #a2 = Evt(tag=args.tag, src=args.src, det=args.det, pfx="source", seqs=[], args=args)
    #print("a2")
    #print(a2)
 

    b = Evt(tag="-%s"%args.tag, src=args.src, det=args.det, pfx=args.pfx, seqs=[], args=args)
    print("b")
    print(b)

    print("a")
    a.history_table(slice(0,5))
    #print("a2")
    #a2.history_table(slice(0,5))

    print("b")
    b.history_table(slice(0,5))
       

