#!/usr/bin/env python
"""
QPropTest.py
===============

::

    ./QPropTest.sh ana 

"""
import logging 
log = logging.getLogger(__name__)
import matplotlib.pyplot as plt 
import os, numpy as np
from opticks.ana.fold import Fold


class QPropTest(object):
    def __init__(self, f ):
        reldir = os.path.basename(f.base)  
        types = {"float":np.float32, "double":np.float64 }

        assert reldir in types
        dtype = f.prop.dtype
        assert types[reldir] == dtype 

        utype = np.uint32 if dtype == np.float32 else np.uint64

        self.f = f 
        self.dtype = dtype
        self.utype = utype
        self.reldir = reldir 
        self.title = "qudarap/tests/QPropTest.py %s " % f.base

    def plot(self):
        f = self.f
        fig, ax = plt.subplots(figsize=[12.8,7.2])
        fig.suptitle(self.title)
        for i in range(len(f.prop)):
            lp = f.prop.view(self.utype)[i,-1,-1]
            ax.scatter( f.prop[i,:lp,0], f.prop[i,:lp,1], label="src-%d-lp-%d" % (i,lp) )
            ax.plot( f.domain, f.lookup[i], label="dst-%d" % i )
        pass
        ax.legend()
        fig.show()
        path = os.path.join(f.base, "fig.png")
        log.info("save to %s " % path)
        fig.savefig(path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    f = Fold.Load("$FOLD/float", symbol="f")
    t = QPropTest(f)  
    t.plot()



