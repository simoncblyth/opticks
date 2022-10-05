#!/usr/bin/env python
"""
U4Debug.py
============

Comparing C and S steps between opticksMode

opticksMode:0
    as if there is no Opticks there

opticksMode:1
    only GPU propagation, CPU photon generation is skipped
    (NB because this changes the Geant4 random consumption the C+S steps will
    not match those obtained wih opticksMode 0)

opticksMode:3
    both CPU and GPU propagations, the CPU ones should be the same as opticksMode:0
    (NB because the Geant4 random consumption of the C+S steps is unchanged
    there should be an exact match between opticksMode 0 and 3)


"""
import os, numpy as np
from opticks.ana.fold import Fold

class U4Debug(object):
    @classmethod
    def Create(cls, rel, base, symbol):
        f = Fold.Load(base, rel, symbol=symbol)
        return None if f is None else cls(f, rel=rel)        

    def __init__(self, f, rel):
        self.rel = rel
        self.f = f 

        c = f.U4Cerenkov_Debug.reshape(-1,8) 
        s = f.U4Scintillation_Debug.reshape(-1,8) 
        h = f.U4Hit_Debug.reshape(-1,4) 
        g = f.gsl.reshape(-1,4) if hasattr(getattr(f,'gsl', None),'shape') else None

        self.c = c 
        self.s = s 
        self.h = h 
        self.g = g

    def __str__(self):
        f = self.f
        return repr(f)

    def __repr__(self):
        rel = self.rel
        c = self.c 
        s = self.s 
        h = self.h 
        g = self.g 
        gs = str(g.shape) if not g is None else "-"
        return "%20s c %10s s %10s h %10s g %10s " % (rel, str(c.shape), str(s.shape), str(h.shape), gs) 
 

if __name__ == '__main__':
    base = "/tmp/u4debug"

    x00 = U4Debug.Create("ntds0/000", symbol="x00", base=base)
    x01 = U4Debug.Create("ntds0/001", symbol="x01", base=base)

    x10 = U4Debug.Create("ntds1/000", symbol="x10", base=base)
    x11 = U4Debug.Create("ntds1/001", symbol="x11", base=base)
 
    x30 = U4Debug.Create("ntds3/000", symbol="x30", base=base)
    x31 = U4Debug.Create("ntds3/001", symbol="x31", base=base)
     
    if not x00 is None and not x30 is None:
        assert np.all( x00.c == x30.c )
        assert np.all( x00.s == x30.s )
    else:
        print("one of x00 or x30 is None")
    pass
    if not x01 is None and not x31 is None:
        assert np.all( x01.c == x31.c )
        assert np.all( x01.s == x31.s )
    else:
        print("one of x01 or x31 is None")
    pass
    ## hit records dont match as gensteps are not collected for opticksMode 0 changing hit indices
    #assert np.all( x00.h == x30.h )  
    #assert np.all( x01.h == x31.h )
    pass


