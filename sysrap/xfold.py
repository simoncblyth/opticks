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
        :return symbol: "a" or "b"
        """
        cx = xf.base.find("CX") > -1  
        u4 = xf.base.find("U4") > -1  
        assert cx ^ u4 
        symbol = "a" if cx else "b"
        return symbol

    @classmethod
    def Ident(cls, x):
        bsym = cls.BaseSymbol(x)
        ident = cls.tag if bsym == "a" else cls.stack
        return ident 

    def __init__(self, x, symbol=None):
        """
        :param x: Fold instance
        """
        t = stag.Unpack(x.tag) if hasattr(x,"tag") else None
        f = getattr(x, "flat", None)
        n = stag.NumStarts(t) if not t is None else None
        ts,fs = stag.StepSplit(t,x.flat) if not t is None else None
        ident = self.Ident(x)        

        self.x = x       # Fold instance, called x because it is usually "a" or "b"
        self.t = t       # (num_photon, SLOTS)  : unpacked consumption tag/stack enumeration integers
        self.f = f       # (num_photon, SLOTS)  : uniform rand 
        self.n = n       # (num_photon,)        : number of steps 
        self.ts = ts     # (num_photon, 7, 10)  : 7 is example max_steps which should match A.n.max(), 10 is example max rand per step input  
        self.fs = fs     # (num_photon, 7, 10)  : flat random values split by steps  

        self.ident = ident  # A:stag OR B:U4Stack instance for providing names to enumeration codes
        self.symbol = symbol
        self.idx = 0 

    def __call__(self, idx):
        self.idx = idx 
        return self

    def header(self):
        idx = self.idx 
        seqhis = seqhis_(self.x.seq[idx,0])
        return "%s(%d) : %s" % (self.symbol, idx, seqhis)

    def __repr__(self):
        return "\n".join([self.header(), self.ident.label(self.t[self.idx],self.f[self.idx])]) 







if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
 
    a = Fold.Load("$A_FOLD", symbol="a")
    b = Fold.Load("$B_FOLD", symbol="b")
    A = XFold(a, symbol="A")
    B = XFold(b, symbol="B")


