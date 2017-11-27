#!usr/bin/env python
"""

"""
import os, sys, logging, numpy as np

class Dv(object):
   def __init__(self, idx, sel, av, bv, lcu, msg=""):
       dv = np.abs( av - bv )
       

       self.idx = idx 
       self.sel = sel 
       self.av = av
       self.bv = bv
       self.dv = dv
       self.lcu  = lcu
       self.msg = msg

   def __repr__(self):
       dv = self.dv
       nto = len(dv)
       label = " %0.4d %10s : %30s : %7d  %7d " % ( self.idx, self.msg, self.sel, self.lcu[1], self.lcu[2] )
       if nto>0:
           ndv = len(dv[dv>0]) 
           mx = dv.max()
           mn = dv.min()
           fdv = float(ndv)/float(nto) 
           av = dv.sum()/float(nto)
           desc =  "  %7d/%7d:%6.3f  mx/mn/av %6.4f/%6.4f/%6.4g  " % ( nto, ndv, fdv, mx,mn,av )
       else:
           fdv = 0.
           av = 0 
           desc = ""
       pass
       return "%s : %s  " % (label, desc )


class DvTab(object):
    def __init__(self, name, seqtab, ab):
        self.name = name
        self.seqtab = seqtab
        self.ab = ab 
        self.dirty = False

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
        else:
            assert self.name
        pass 
        assert ab.a.sel == ab.b.sel 
        sel = ab.a.sel 
        dv = Dv(i, sel, av, bv, lcu)
        return dv if len(dv.dv) > 0 else None

    def __repr__(self):
        return "\n".join( [self.name] + map(repr, filter(None,self.dvs) ))



if __name__ == '__main__':
    pass
pass





