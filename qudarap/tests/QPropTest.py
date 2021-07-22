#!/usr/bin/env python
"""
::
 
    ipython -i QPropTest.py 


"""
import matplotlib.pyplot as plt 
import os, numpy as np


class QPropTest(object):
    FOLD = "/tmp/QPropTest"
    def __init__(self):
        domain = np.load(os.path.join(self.FOLD, "domain.npy"))
        lookup = np.load(os.path.join(self.FOLD, "lookup.npy"))
        prop = np.load(os.path.join(self.FOLD, "prop.npy"))
        
        self.domain = domain
        self.lookup = lookup
        self.prop = prop


if __name__ == '__main__':
    t = QPropTest()  

    utype = np.uint32

    fig, ax = plt.subplots(figsize=[12.8,7.2])
    for i,p in enumerate(t.prop):
        lp = p.view(utype)[-1]
        i2 = p.view(utype)[-2]
        assert i2 == i
        pr = p.reshape(-1,2)
        ax.scatter( pr[:lp,0], pr[:lp,1], label="src-%d-lp-%d" % (i,lp) )
        ax.plot(  t.domain, t.lookup[i], label="dst-%d" % i )
    pass
    ax.legend()
    fig.show()




