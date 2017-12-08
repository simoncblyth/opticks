#!/usr/bin/env python
"""
tboolean.py 
=============================================




::

    ab.sel = "[TO] BT BT BT BT SA" 
    hh = ab.hh



    In [21]: from opticks.ana.histype import HisType

    In [22]: histype = HisType()

    In [25]: histype.code("TO BT AB")
    Out[25]: 1229

    In [26]: ab.a.seqhis
    Out[26]: 
    A()sliced
    A([36045, 36045,  2237, ..., 36045, 36045, 36045], dtype=uint64)

    In [27]: ab.a.seqhis.shape
    Out[27]: (100000,)

    In [29]: np.where(ab.a.seqhis == histype.code("TO BT AB"))[0]
    Out[29]: array([ 2084,  4074, 15299, 20870, 25748, 26317, 43525, 51563, 57355, 61602, 65894, 71978, 77062, 78744, 79117, 86814])

    In [30]: np.where(ab.a.seqhis == histype.code("TO BT AB"))[0].shape
    Out[30]: (16,)


    In [36]: ab.a.dindex("TO BT AB")
    Out[36]: '--dindex=2084,4074,15299,20870,25748,26317,43525,51563,57355,61602'

    In [37]: ab.b.dindex("TO BT AB")
    Out[37]: '--dindex=2084,4074,15299,20870,25748,26317,43525,51563,57355,61602'




"""
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.ab   import AB

if __name__ == '__main__':
    ok = opticks_main(doc=__doc__, tag="1", src="torch", det="tboolean-torus", smry=False)  

    #print "ok.smry %d " % ok.smry 
    log.info(ok.brief)

    ab = AB(ok)

    print ab
    print "ab.a.metadata", ab.a.metadata
    print "ab.a.metadata.csgmeta0", ab.a.metadata.csgmeta0

    print ab.rpost_dv
    print ab.rpol_dv
    print ab.ox_dv


    rc = ab.RC 

    if not ok.ipython:
        log.info("early exit as non-interactive")
        sys.exit(rc)
    else:
        pass
    pass

    a = ab.a
    b = ab.b
    #ab.aselhis = "TO BT BT SA"     # dev aligned comparisons
  
   

    

