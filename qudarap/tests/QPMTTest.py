#!/usr/bin/env python

import os, numpy as np, matplotlib as mp
from opticks.ana.fold import Fold

SIZE = np.array([1280, 720]) 

class QPMTTest(object):
    def __init__(self):
        pass 

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))



    prop_ni = t.rindex[:,-1,-1].view(np.int32)  

    e = t.domain

    e0 = 2.3
    e1 = 3.3
    s = np.logical_and( e >= e0, e <= e1 ) 

    ni = t.interp.shape[0]
    nj = t.interp.shape[1]
    nk = t.interp.shape[2]



    fig, axs = mp.pyplot.subplots(ni, figsize=SIZE/100.)


    for i in range(ni):

        ax = axs[i]
        for j in range(nj):
            for k in range(nk):
                v = t.interp[i,j,k]  
                iprop = i*nj*nk + j*nk + k 

                label = "iprop %d pmtcat:%d layr:%d prop:%d" % (iprop, i,j,k) 
                ax.plot( e[s], v[s], label=label ) 

                p_ni = prop_ni[iprop]
                p_e = t.rindex[iprop,:p_ni,0] 
                p_v = t.rindex[iprop,:p_ni,1] 

                p_s = np.logical_and( p_e >= e0, p_e <= e1 )
                ax.scatter( p_e[p_s], p_v[p_s] )
            pass
        pass
    pass
    # upper/center/lower right/left 
    #ax.legend(loc=os.environ.get("LOC", "lower center"))
    fig.show()


