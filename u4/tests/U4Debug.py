#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':

    base = "/tmp/u4debug"

    zero = False
    three = True
    
    if zero:
        f00 = Fold.Load(base, "ntds0/000", symbol="f00" ) 
        f01 = Fold.Load(base, "ntds0/001", symbol="f01" ) 
        f00c = f00.U4Cerenkov_Debug.reshape(-1,8) 
        f01c = f01.U4Cerenkov_Debug.reshape(-1,8) 
        f00s = f00.U4Scintillation_Debug.reshape(-1,8) 
        f01s = f01.U4Scintillation_Debug.reshape(-1,8) 
        f00h = f00.U4Hit_Debug.reshape(-1,4) 
        f01h = f01.U4Hit_Debug.reshape(-1,4) 
        print("f00c %10s f00s %10s f00h %10s " % (str(f00c.shape), str(f00s.shape), str(f00h.shape)) )
        print("f01c %10s f01s %10s f01h %10s " % (str(f01c.shape), str(f01s.shape), str(f01h.shape)) )
    pass

    if three:
        f30 = Fold.Load(base, "ntds3/000", symbol="f30" ) 
        f31 = Fold.Load(base, "ntds3/001", symbol="f31" ) 
        f30c = f30.U4Cerenkov_Debug.reshape(-1,8) 
        f31c = f31.U4Cerenkov_Debug.reshape(-1,8) 
        f30s = f30.U4Scintillation_Debug.reshape(-1,8) 
        f31s = f31.U4Scintillation_Debug.reshape(-1,8) 
        f30h = f30.U4Hit_Debug.reshape(-1,4) 
        f31h = f31.U4Hit_Debug.reshape(-1,4) 
        print("f30c %10s f30s %10s f30h %10s " % (str(f30c.shape), str(f30s.shape), str(f30h.shape)) )
        print("f31c %10s f31s %10s f31h %10s " % (str(f31c.shape), str(f31s.shape), str(f31h.shape)) )
    pass
     
    if zero and three:
        # opticksMode 0<->3 should match : currently relying on Cerenkov pinning 
        assert np.all( f00c == f30c )
        assert np.all( f01c == f31c )
        assert np.all( f00s == f30s )
        assert np.all( f01s == f31s )
        assert np.all( f00h == f30h )
        assert np.all( f01h == f31h )
    pass




