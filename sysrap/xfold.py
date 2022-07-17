#!/usr/bin/env python 

import numpy as np
import os, re, logging
log = logging.getLogger(__name__)

from opticks.ana.fold import Fold
from opticks.sysrap.stag import stag 
from opticks.u4.U4Stack import U4Stack 
from opticks.ana.p import *


class XFold(object):
    tag = stag()
    stack = U4Stack()

    @classmethod
    def BaseSymbol(cls, xf):
        """  
        :param xf: Fold instance
        :return symbol: "A" or "B" : A for Opticks, B for Geant4  
        """
        CX = xf.base.find("CX") > -1  
        U4 = xf.base.find("U4") > -1  
        assert CX ^ U4  # exclusive-OR
        symbol = "A" if CX else "B"
        return symbol

    @classmethod
    def Ident(cls, x):
        bsym = cls.BaseSymbol(x)
        ident = cls.tag if bsym == "A" else cls.stack
        return ident 

    def __init__(self, x, symbol=None):
        """
        :param x: Fold instance, eg either a or b 
        """
        t = stag.Unpack(x.tag) if hasattr(x,"tag") else None
        f = getattr(x, "flat", None)
        n = stag.NumStarts(t) if not t is None else None
        log.info("XFold before stag.StepSplit")
        ts,fs = stag.StepSplit(t,x.flat) if not t is None else None
        log.info("XFold after stag.StepSplit")
        ident = self.Ident(x)        

        xsymbol = self.BaseSymbol(x)
        if xsymbol == "B":
            t2 = self.stack.make_stack2tag_mapped(t) 
            ts2 = self.stack.make_stack2tag_mapped(ts) 
        elif xsymbol == "A":
            t2 = self.stack.make_tag2stack_mapped(t) 
            ts2 = self.stack.make_tag2stack_mapped(ts) 
        else:
            assert 0
        pass 
        if symbol != xsymbol:
            log.error("using unconventional symbol %s xsymbol %s " % (symbol, xsymbol))
        pass 
        #assert symbol == xsymbol 

        self.x = x       # Fold instance, called x because it is usually "a" or "b"
        self.t = t       # (num_photon, SLOTS)  : unpacked consumption tag/stack enumeration integers
        self.f = f       # (num_photon, SLOTS)  : uniform rand 
        self.n = n       # (num_photon,)        : number of steps 
        self.t2 = t2     # (num_photon, SLOTS)  : native enumeration mapped to the other enumeration

        self.ts = ts     # (num_photon, 7, 10)  : 7 is example max_steps which should match A.n.max(), 10 is example max rand per step input  
        self.fs = fs     # (num_photon, 7, 10)  : flat random values split by steps  
        self.ts2 = ts2   # (num_photon, 7, 10)  : native enumeration mapped to the other enumeration 

        self.ident = ident  # A:stag OR B:U4Stack instance for providing names to enumeration codes
        self.symbol = symbol
        self.xsymbol = xsymbol
        self.idx = 0 
        self.flavor = ""

    def __call__(self, idx):
        self.idx = idx 
        self.flavor = "call"
        return self

    def __getitem__(self, idx):
        self.idx = idx 
        self.flavor = "getitem"
        return self

    def header(self):
        idx = self.idx 
        seqhis = seqhis_(self.x.seq[idx,0])
        return "%s(%d) : %s" % (self.symbol, idx, seqhis)

    def rbnd__(self):
        return boundary___(self.x.record[self.idx])
    def rori__(self):
        return orient___(self.x.record[self.idx]) 
    def rpri__(self):
        return primIdx___(self.x.record[self.idx])   
    def rins__(self):
        return instanceId__(self.x.record[self.idx])   


    def rbnd_(self): 
        bb_ = self.rbnd__()
        oo_ = self.rori__()
        pp_ = self.rpri__()
        ii_ = self.rins__()
        assert bb_.shape == oo_.shape
        assert bb_.shape == pp_.shape
        assert bb_.shape == ii_.shape

        wp = np.where( bb_ > 0 )  # mask boundary zero to skip unsets 
        bb = bb_[wp] 
        oo = oo_[wp] 
        pp = pp_[wp]
        ii = ii_[wp]

        pm = ["+","-"]
        lines = [ "%s %-40s %-50s"%(pm[oo[i]],cf.sim.bndnamedict.get(bb[i]), cf.primIdx_meshname_dict.get(pp[i])) for i in range(len(bb))]

        lines += [""]
        lines += ["- : against the normal (ie inwards from omat to imat)"]
        lines += ["+ : with the normal (ie outwards from imat to omat)"] 
        return lines 


    def rbnd(self):
        return "\n".join(self.rbnd_())

    def content(self):
        lines = []
        members = "t t2 n ts fs ts2".split()
        for mem in members:
            arr = getattr(self, mem) 
            name = "%s.%s" % ( self.symbol, mem)
            line = "%10s : %s " % (name, str(arr.shape))
            lines.append(line)
        pass
        return "\n".join(lines)

    def body(self):
        return self.ident.label(self.t[self.idx],self.f[self.idx])

    def identification(self):
        return "%s : %s" % (self.symbol, self.x.base) 

    def call_repr(self):
        return "\n".join([self.header(), self.content(), self.body()]) 

    def getitem_repr(self):
        return "\n".join([self.header(), self.rbnd()])

    def __repr__(self):
        if self.flavor == "call":
            rep = self.call_repr()
        elif self.flavor == "getitem":
            rep = self.getitem_repr()
        else:
            rep = self.call_repr()
        pass
        return rep 




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
 
    a = Fold.Load("$A_FOLD", symbol="a")
    b = Fold.Load("$B_FOLD", symbol="b")
    A = XFold(a, symbol="A")
    B = XFold(b, symbol="B")


