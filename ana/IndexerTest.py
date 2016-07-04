#!/usr/bin/env python
"""
optickscore-/tests/IndexerTest 
================================

GPU and CPU indices mismatch
-----------------------------

Maybe a stable sort type of issue.

In any case discrepancies are all in the low frequency of occurence tail, 
so probably acceptable.

::

    seqhis
    [[ 28  29]
     [ 22  23]
     [ 23  22]
     [ 32 255]
     [ 23  22]
     [ 28  29]
     [255  32]
     [ 31  30]
     [ 22  23]
     [ 23  22]
     [ 31  30]
     [ 23  22]
     [ 23  22]
     [ 22  23]
     [ 22  23]
     [ 31  30]
     [ 29  31]
     [ 23  22]
     [ 32 255]
     [ 22  23]
     [255  32]
     [ 22  23]
     [ 28  29]
     [ 22  23]
     [ 23  22]
     [ 29  31]
     [ 30  28]
     [ 30  28]
     [ 30  28]
     [ 29  31]]

"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.ana import Evt

def compare(a, b):
    return np.vstack([a,b]).T[a != b]


if __name__ == '__main__':

    typ = "torch" 
    tag = "4" 
    det = "dayabay" 
    cat = "PmtInBox"  

    evt = Evt(tag=tag, src=typ, det=cat)
    print evt
    ps = np.load("/tmp/phosel.npy")

    print "seqhis\n", compare(evt.ps[:,0,0], ps[:,0,0])
    print "seqmat\n", compare(evt.ps[:,0,1], ps[:,0,1])

    


