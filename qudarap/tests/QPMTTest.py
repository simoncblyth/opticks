#!/usr/bin/env python

import os, numpy as np, matplotlib as mp
from opticks.ana.fold import Fold

SIZE = np.array([1280, 720]) 

class QPMTTest(object):
    def __init__(self):
        pass 

if __name__ == '__main__':
    t = Fold.Load("/tmp/QPMTTest", symbol="t")
    #t = Fold.Load("/tmp/QPMTTest/rindex", symbol="t")
    print(repr(t))

    fig, ax = mp.pyplot.subplots(figsize=SIZE/100.)

    #sli = slice(50,200)
    sli = slice(None)

    prop_ni = t.rindex[:,-1,-1].view(np.int32)  

    e = t.domain

    ni = t.interp.shape[0]
    nj = t.interp.shape[1]
    nk = t.interp.shape[2]

    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                v = t.interp[i,j,k]  
                iprop = i*nj*nk + j*nk + k 

                label = "iprop %d pmtcat:%d layr:%d prop:%d" % (iprop, i,j,k) 
                ax.plot( e[sli], v[sli], label=label ) 

                p_ni = prop_ni[iprop]
                p_e = t.rindex[iprop,:p_ni,0] 
                p_v = t.rindex[iprop,:p_ni,1] 

                ax.scatter( p_e, p_v )
            pass
        pass
    pass
    # upper/center/lower right/left 
    #ax.legend(loc=os.environ.get("LOC", "lower center"))
    fig.show()


