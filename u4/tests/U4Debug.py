#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':

    base = "/tmp/u4debug"
    f00 = Fold.Load(base, "ntds0/000", symbol="f00" ) 
    f01 = Fold.Load(base, "ntds0/001", symbol="f01" ) 
    f30 = Fold.Load(base, "ntds3/000", symbol="f30" ) 
    f31 = Fold.Load(base, "ntds3/001", symbol="f31" ) 

    #print(repr(f00))
    #print(repr(f01))
    #print(repr(f30))
    #print(repr(f31))

    f00c = f00.U4Cerenkov_Debug.reshape(-1,8) 
    f01c = f01.U4Cerenkov_Debug.reshape(-1,8) 
    f30c = f30.U4Cerenkov_Debug.reshape(-1,8) 
    f31c = f31.U4Cerenkov_Debug.reshape(-1,8) 

    f00s = f00.U4Scintillation_Debug.reshape(-1,8) 
    f01s = f01.U4Scintillation_Debug.reshape(-1,8) 
    f30s = f30.U4Scintillation_Debug.reshape(-1,8) 
    f31s = f31.U4Scintillation_Debug.reshape(-1,8) 

    print("f00c %10s f00s %10s " % (str(f00c.shape), str(f00s.shape)) )
    print("f01c %10s f01s %10s " % (str(f01c.shape), str(f01s.shape)) )
    print("f30c %10s f30s %10s " % (str(f30c.shape), str(f30s.shape)) )
    print("f31c %10s f31s %10s " % (str(f31c.shape), str(f31s.shape)) )

    # opticksMode 0<->3 should match : currently relying on Cerenkov pinning 
    assert np.all( f00c == f30c )
    assert np.all( f01c == f31c )
    assert np.all( f00s == f30s )
    assert np.all( f01s == f31s )


