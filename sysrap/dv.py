#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.array_repr_mixin import ArrayReprMixin

class AlignedDV(ArrayReprMixin, object):
    """
    Simple deviation comparisons of random aligned arrays 

    In [2]: pdv.dv
    Out[2]: 
    array([[   47,   117,  1732,  4412,  2710,   965,    16,     1,     0,     0],
           [ 2746,  5430,  1724,    96,     4,     0,     0,     0,     0,     0],
           [ 6404,  2937,   647,    11,     1,     0,     0,     0,     0,     0],
           [ 9995,     1,     1,     3,     0,     0,     0,     0,     0,     0],
           [10000,     0,     0,     0,     0,     0,     0,     0,     0,     0]], dtype=uint32)

    """
    EDGES = np.array( [0.,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], dtype=np.float32 ) 
    COLUMN_LABEL = ["%g" % _ for _ in EDGES[1:]]

    POS = 0 
    TIME = 1 
    MOM = 2 
    POL = 3
    WL = 4 

    ITEMS = (POS, TIME, MOM, POL, WL)
    ROW_LABEL = ("pos", "time", "mom", "pol", "wl")

    def __init__(self, a, b, arr="photon", symbol=None):
        """ 
        :param a: opticks.ana.fold Fold instance
        :param b: opticks.ana.fold Fold instance

        In [12]: adv[...,0,3].shape
        Out[12]: (10000,)

        In [13]: bdv[...,0,3].shape
        Out[13]: (10000, 10)
        """

        wseq = np.where( a.seq[:,0] == b.seq[:,0] ) 
        adv = np.abs( getattr(a,arr)[wseq] - getattr(b,arr)[wseq] )  ## for deviations to be meaningful needs to be same history  


        ## although ... ellipsis indexing can generalize access between photon and record 
        ## need to use different np.amax axis args to keep the deviation at photon level, not record level 
        if arr == "photon":  
            time = adv[:,0,3] 
            pos  = np.amax( adv[:,0,:3], axis=1 )  ## amax of the 3 position deviations, so can operate at photon position level, not x,y,z level 
            mom  = np.amax( adv[:,1,:3], axis=1 )
            pol  = np.amax( adv[:,2,:3], axis=1 )
            wl   = adv[:,2,3]
        elif arr == "record":
            time = np.amax( adv[:,:,0,3], axis=1 ) 
            pos  = np.amax( adv[:,:,0,:3], axis=(1,2) )  ## amax of the 3 position deviations, so can operate at photon position level, not x,y,z level 
            mom  = np.amax( adv[:,:,1,:3], axis=(1,2) )
            pol  = np.amax( adv[:,:,2,:3], axis=(1,2) )
            wl   = np.amax( adv[:,:,2,3], axis=1 )
        else:
            assert 0, "expecting photon or record not %s " % arr
        pass

        dv = np.zeros( (len(self.ITEMS),len(self.EDGES)-1), dtype=np.uint32 )
        dv[self.POS], bins = np.histogram( pos, bins=self.EDGES ) 
        dv[self.TIME], bins = np.histogram( time, bins=self.EDGES ) 
        dv[self.MOM], bins = np.histogram( mom, bins=self.EDGES ) 
        dv[self.POL], bins = np.histogram( pol, bins=self.EDGES ) 
        dv[self.WL], bins = np.histogram( wl, bins=self.EDGES ) 

        self.symbol = symbol if not symbol is None else "%s.dv" % arr
        self.pos = pos
        self.time = time
        self.mom = mom 
        self.pol = pol
        self.wl = wl 
        self.dv = dv

    def __repr__(self):
        return self.MakeRepr(self.dv, symbol=self.symbol)


class HeaderAB(object):
    def __init__(self, a, b, cmdline):
        self.lines = ["A_FOLD : %s " % a.base, "B_FOLD : %s " % b.base, cmdline ]
    def __repr__(self):
        return "\n".join(self.lines) 


if __name__ == '__main__':
    a = Fold.Load("$A_FOLD", symbol="a") if "A_FOLD" in os.environ else None
    b = Fold.Load("$B_FOLD", symbol="b") if "B_FOLD" in os.environ else None

    hdr = HeaderAB(a,b, "./dv.sh   # cd ~/opticks/sysrap")
    print(hdr)

    pdv = AlignedDV(a,b, arr="photon", symbol="pdv")
    print(pdv)    

    rdv = AlignedDV(a,b, arr="record", symbol="rdv")
    print(rdv)    



