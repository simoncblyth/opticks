#!/usr/bin/env python
"""
"""
import os, logging
import numpy as np
import matplotlib.pyplot as plt

from env.numerics.npy.metadata import Metadata, Catdir


def speedplot(cat, tag, a, landscape=False, ylim=None, log_=False):
    nnp = len(np.unique(a.numphotons))
    assert nnp == 1, "Tags and negated counterparts should always have the same photon statistics" 

    mega = float(a.numphotons[0])/1e6
    title = "Propagate times (s) for %3.1fM Photons with %s geometry, tag %s, [max/avg/min]" % (mega, cat, tag)  

    plt.close()
    plt.ion()

    fig = plt.figure()
    fig.suptitle(title)

    compute = a.flgs & Metadata.COMPUTE != 0 
    interop = a.flgs & Metadata.INTEROP != 0 
    cfg4    = a.flgs & Metadata.CFG4 != 0 

    msks = [cfg4, interop, compute]
    ylims = [[0,60],[0,5],[0,1]]
    labels = ["CfGeant4", "Opticks Interop", "Opticks Compute"]

    n = len(msks)
    for i, msk in enumerate(msks):

        if landscape: 
            ax = fig.add_subplot(1,n,i+1)
        else:
            ax = fig.add_subplot(n,1,i+1)
        pass
        d = a[msk]

        t = d.propagate

        mn = t.min()
        mx = t.max()
        av = np.average(t)        

        label = "%s [%5.2f/%5.2f/%5.2f] " % (labels[i], mx,av,mn)
 
        loc = "lower right" if i == 0 else "upper right" 

        ax.plot( d.index, d.propagate, "o")
        ax.plot( d.index, d.propagate, drawstyle="steps", label=label)

        if log_:
            ax.set_yscale("log")

        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(ylims[i])
        pass
        ax.legend(loc=loc)
    pass


    ax.set_xlabel('All times from: MacBook Pro (2013), NVIDIA GeForce GT 750M 2048 MB (384 cores)')
    ax.xaxis.set_label_coords(-0.5, -0.07 )

    plt.show()




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)    

    #cat, tag = "rainbow", "6"
    cat, tag = "PmtInBox", "4"
    
    catd = Catdir(cat)
    a = catd.times(tag)


if 1:
    speedplot(cat, tag, a, landscape=True, ylim=[0.1, 60], log_=True)
    

