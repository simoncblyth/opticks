#!/usr/bin/env python

import os, numpy as np, builtins
from opticks.ana.fold import Fold

BASE = os.environ["BASE"]

class AA(object):
    def __init__(self, path_=lambda i:"%d.npy" % i):
        symbols = "abcdefghijklmnopqrstuvwxyz" 
        lines = []
        for i in range(10):
            path = path_(i)
            if not os.path.exists(path): continue 
            symbol = symbols[i]
            a = np.load(path)
            setattr(builtins, symbol, a ) 
            msg = "symbol %s a %20s path %s " % (symbol, str(a.shape), path)
            lines.append(msg)
        pass    
        self.lines = lines 

    def __repr__(self):
        return "\n".join(self.lines)

if __name__ == '__main__':
    path_ = lambda i:os.path.join(BASE, "GGeo/GMergedMesh/%d/placement_iidentity.npy" % i) 
    aa = AA( path_ ); 
    print(aa)


    sidx = np.concatenate( [b[:,2,3], c[:,5,3], d[:,5,3], e[:,4,3] ] )  
    ssidx = sidx[np.argsort(sidx)]  
    xsidx = np.arange( len(ssidx), dtype=np.uint32 ) 
    assert np.all( ssidx - 1 == xsidx )


    t = Fold.Load(symbol="t")
    print(repr(t))

    i = t.inst.view(np.int64) 
    w2 = np.where( i[:,1,3] == 2 )[0]  
    w3 = np.where( i[:,1,3] == 3 )[0]  
    w4 = np.where( i[:,1,3] == 4 )[0]  
    w5 = np.where( i[:,1,3] == 5 )[0]  
    w6 = np.where( i[:,1,3] == 6 )[0]  
    









