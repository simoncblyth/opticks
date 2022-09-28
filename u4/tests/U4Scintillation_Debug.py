#!/usr/bin/env python

import numpy as np

if __name__ == '__main__':


    a00p = "/tmp/ntds0/000/U4Scintillation_Debug.npy"
    a01p = "/tmp/ntds0/001/U4Scintillation_Debug.npy"
    a30p = "/tmp/ntds3/000/U4Scintillation_Debug.npy"
    a31p = "/tmp/ntds3/001/U4Scintillation_Debug.npy"

    a00 = np.load(a00p).reshape(-1,8)
    a01 = np.load(a01p).reshape(-1,8)
    a30 = np.load(a30p).reshape(-1,8)
    a31 = np.load(a31p).reshape(-1,8)

    print("a00 %s %s " % (a00p,str(a00.shape)))
    print("a01 %s %s " % (a01p,str(a01.shape)))
    print("a30 %s %s " % (a30p,str(a30.shape)))
    print("a31 %s %s " % (a31p,str(a31.shape)))




    b00p = "/tmp/scintcheck/ntds0/000/U4Scintillation_Debug.npy"
    b01p = "/tmp/scintcheck/ntds0/001/U4Scintillation_Debug.npy"
    b30p = "/tmp/scintcheck/ntds3/000/U4Scintillation_Debug.npy"
    b31p = "/tmp/scintcheck/ntds3/001/U4Scintillation_Debug.npy"

    b00 = np.load(b00p).reshape(-1,8)
    b01 = np.load(b01p).reshape(-1,8)
    b30 = np.load(b30p).reshape(-1,8)
    b31 = np.load(b31p).reshape(-1,8)

    print("b00 %s %s " % (b00p,str(b00.shape)))
    print("b01 %s %s " % (b01p,str(b01.shape)))
    print("b30 %s %s " % (b30p,str(b30.shape)))
    print("b31 %s %s " % (b31p,str(b31.shape)))

    # reproducibility between separate running 
    #assert np.all( a00 == b00 )
    #assert np.all( a01 == b01 )
    #assert np.all( a31 == b31 )
    #assert np.all( a30 == b30 )



