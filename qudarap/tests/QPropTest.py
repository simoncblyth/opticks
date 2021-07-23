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
    def __init__(self):
        prop = np.load(os.path.join(self.FOLD, "prop.npy"))
        domain = np.load(os.path.join(self.FOLD, "domain.npy"))
        lookup = np.load(os.path.join(self.FOLD, "lookup.npy"))
      
        dtype = prop.dtype
        assert dtype in [np.float32, np.float64]
        utype = np.uint32 if dtype == np.float32 else np.uint64

        self.dtype = dtype
        self.utype = utype
        self.prop = prop
        self.domain = domain
        self.lookup = lookup

    def plot(self):
        t = self
        fig, ax = plt.subplots(figsize=[12.8,7.2])
        for i,p in enumerate(t.prop):
            lp = p.view(t.utype)[-1]
            i2 = p.view(t.utype)[-2]
            assert i2 == i
            pr = p.reshape(-1,2)
            ax.scatter( pr[:lp,0], pr[:lp,1], label="src-%d-lp-%d" % (i,lp) )
            ax.plot(  t.domain, t.lookup[i], label="dst-%d" % i )
        pass
        ax.legend()
        fig.show()
        path = os.path.join(t.FOLD, "fig.png")
        log.info("save to %s " % path)
        fig.savefig(path)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    t = QPropTest()  
    t.plot()



