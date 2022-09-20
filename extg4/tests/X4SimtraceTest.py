#!/usr/bin/env python
"""
X4SimtraceTest.py
===================

"""
import os, logging, builtins, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as mp
from opticks.ana.fold import Fold
from opticks.sysrap.sframe import sframe , X, Y, Z
from opticks.ana.eget import efloatarray_

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)


    SYMBOLS = os.environ.get("SYMBOLS", None)
    FOLD = os.environ.get("FOLD", None)

    if SYMBOLS is None and not FOLD is None:
        s = Fold.Load(symbol="s")
        t = None
        fr = s.sframe
        s_label = os.environ["GEOM"]
        SYMBOLS="S" 
    elif not SYMBOLS is None:
        ff = Fold.MultiLoad()
        frs = list(filter(None, map(lambda f:f.sframe, ff)))
        assert len(frs) > 0
        fr = frs[0]       ## HMM: picking first frame, maybe need to form composite bbox from all frames ?
    else:
        assert 0 
    pass


    fig, ax = fr.mp_subplots(mp)  

    log.info("SYMBOLS %s " % str(SYMBOLS))
    if not SYMBOLS is None:
        for A in list(SYMBOLS):
            a = A.lower()
            log.info("A %s a %s" % ( A, a ) )
            if hasattr(builtins, a):
                fold = getattr(builtins, a)
                label = getattr(builtins, "%s_label" % a )
                label = label.split("__")[0] if "__" in label else label
                a_offset = efloatarray_("%s_OFFSET" % A, "0,0,0")
                log.info("label %s" % ( label ) )
                a_hit = fold.simtrace[:,0,3]>0
                a_pos = a_offset + fold.simtrace[a_hit][:,1,:3]
                fr.mp_scatter(a_pos, label="%s:%s" % (A,label), s=1 )
            else:
                log.info(" a %s not in builtins" % a)
            pass
        pass
    pass
    fr.mp_legend()
    fig.show()



