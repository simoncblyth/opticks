#!usr/bin/env python
"""
Examining the deviation::

    In [15]: ab.rpost_dv
    Out[15]: 
    rpost_dv
     0000            :                          TO SA :   55321    55303  :     55249/      0: 0.000  mx/mn/av 0.0000/0.0000/     0    
     0001            :                    TO BT BT SA :   39222    39231  :     34492/      8: 0.000  mx/mn/av 0.0138/0.0000/3.192e-06    
     0002            :                       TO BR SA :    2768     2814  :       188/      0: 0.000  mx/mn/av 0.0000/0.0000/     0    
     0003            :                 TO BT BR BT SA :    2425     2369  :       125/      0: 0.000  mx/mn/av 0.0000/0.0000/     0    
     0004            :              TO BT BR BR BT SA :     151      142  :         1/      0: 0.000  mx/mn/av 0.0000/0.0000/     0    

    In [12]: dv = ab.rpost_dv.dvs[1].dv
    In [16]: av = ab.rpost_dv.dvs[1].av
    In [17]: bv = ab.rpost_dv.dvs[1].bv

    In [14]: np.where( dv > 0 )
    Out[14]: 
    (
        A([ 8019,  8019,  8019,  8019, 13879, 13879, 13879, 13879]),
        A([0, 1, 2, 3, 0, 1, 2, 3]),
        A([1, 1, 1, 1, 0, 0, 0, 0])
    )

    In [18]: wdv = np.where( dv > 0 )

    In [19]: av[wdv]
    Out[19]: 
    A([  30.4181,   30.4181,   30.4181,   30.4181,  116.2219,  116.2219,  116.2219,  116.2219])

    In [20]: bv[wdv]
    Out[20]: 
    A([  30.4319,   30.4319,   30.4319,   30.4319,  116.2357,  116.2357,  116.2357,  116.2357])

    In [21]: av[wdv] - bv[wdv]
    Out[21]: 
    A([-0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138])


"""
import os, sys, logging, numpy as np

class Dv(object):
   def __init__(self, idx, sel, av, bv, lcu, eps, msg=""):

       dv = np.abs( av - bv )
       
       self.idx = idx 
       self.sel = sel 
       self.av = av
       self.bv = bv
       self.dv = dv
       self.lcu  = lcu
       self.eps = eps
       self.msg = msg

   def __repr__(self):
       dv = self.dv
       nto = len(dv)
       label = " %0.4d %10s : %30s : %7d  %7d " % ( self.idx, self.msg, self.sel, self.lcu[1], self.lcu[2] )
       if nto>0:
           ndv = len(dv[dv>self.eps]) 
           mx = dv.max()
           mn = dv.min()
           fdv = float(ndv)/float(nto) 
           av = dv.sum()/float(nto)
           desc =  "  %7d/%7d:%6.3f  mx/mn/av %6.4g/%6.4g/%6.4g  eps:%g  " % ( nto, ndv, fdv, mx,mn,av, self.eps )
       else:
           fdv = 0.
           av = 0 
           desc = ""
       pass
       return "%s : %s  " % (label, desc )


class DvTab(object):
    def __init__(self, name, seqtab, ab ):
        self.name = name
        self.seqtab = seqtab
        self.ab = ab 
        self.dirty = False
        self.eps = ab.dveps

        labels = self.seqtab.labels
        cu = self.seqtab.cu
        assert len(labels) == len(cu)
        nsel = len(labels)


        dvs = []
        for i in range(nsel):

            sel = labels[i]
            lcu = cu[i]
            assert len(lcu) == 3
            _, na, nb = lcu 

            ab.aselhis = sel 
            ab.checkrec()

            dv = self.dv_(i, sel, lcu)
            dvs.append(dv)
        pass
        self.dvs = dvs 

    def dv_(self, i, sel, lcu):
        ab = self.ab
        if self.name == "rpost_dv": 
            av = ab.a.rpost()
            bv = ab.b.rpost()
        elif self.name == "rpol_dv": 
            av = ab.a.rpol()
            bv = ab.b.rpol()
        elif self.name == "ox_dv": 
            av = ab.a.ox
            bv = ab.b.ox
        else:
            assert self.name
        pass 
        assert ab.a.sel == ab.b.sel 
        sel = ab.a.sel 
        dv = Dv(i, sel, av, bv, lcu, eps=self.eps)
        return dv if len(dv.dv) > 0 else None

    def __repr__(self):
        return "\n".join( [self.name] + map(repr, filter(None,self.dvs) ))



if __name__ == '__main__':
    pass
pass





