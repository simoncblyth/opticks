#!/usr/bin/env python
"""
::
 
    ipython -i QPropTest.py 


"""
import logging 
log = logging.getLogger(__name__)
import matplotlib.pyplot as plt 
import os, numpy as np


class QPropTest(object):
    FOLD = "/tmp/QPropTest"
    def __init__(self, reldir):
        base = os.path.join(self.FOLD, reldir)
        prop = np.load(os.path.join(base, "prop.npy"))
        domain = np.load(os.path.join(base, "domain.npy"))
        lookup = np.load(os.path.join(base, "lookup.npy"))
      
        dtype = prop.dtype

        if reldir == "float":
            assert dtype == np.float32
        elif reldir == "double":
            assert dtype == np.float64
        else:
            assert 0, "unexpected reldir %s " % reldir
        pass

        utype = np.uint32 if dtype == np.float32 else np.uint64

        self.dtype = dtype
        self.utype = utype
        self.prop = prop
        self.domain = domain
        self.lookup = lookup
        self.reldir = reldir 
        self.title = "qudarap/tests/QPropTest.py %s " % base

    def plot(self):
        t = self
        fig, ax = plt.subplots(figsize=[12.8,7.2])
        fig.suptitle(self.title)
        for i in range(len(t.prop)):
            lp = t.prop.view(t.utype)[i,-1,-1]
            ax.scatter( t.prop[i,:lp,0], t.prop[i,:lp,1], label="src-%d-lp-%d" % (i,lp) )
            ax.plot(  t.domain, t.lookup[i], label="dst-%d" % i )
        pass
        ax.legend()
        fig.show()
        path = os.path.join(t.FOLD, self.reldir, "fig.png")
        log.info("save to %s " % path)
        fig.savefig(path)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    #reldir = "float"
    reldir = "double"
    t = QPropTest(reldir)  
    t.plot()

    prop = t.prop
    domain = t.domain
    lookup = t.lookup

