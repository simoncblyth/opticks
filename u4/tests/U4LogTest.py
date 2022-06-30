#!/usr/bin/env python
"""
U4LogTest.py
==============

See notes/issues/U4LogTest_maybe_replacing_G4Log_G4UniformRand_in_Absorption_and_Scattering_with_float_version_will_avoid_deviations.rst

"""
import os, numpy as np


if __name__ == '__main__':

    a_path = os.path.expandvars("/tmp/logTest.npy")   
    b_path = os.path.expandvars("/tmp/$USER/opticks/U4LogTest/scan.npy")

    a = np.load(a_path) if os.path.exists(a_path) else None
    b = np.load(b_path) if os.path.exists(b_path) else None

    if not a is None:print("a %s a_path %s " % (str(a.shape),a_path) )
    if not b is None:print("b %s b_path %s " % (str(b.shape),b_path) )

    U,D0,F0,D4,F4 = range(5)    

    print(a)
    print(b)

    assert np.all( a[:,U] == b[:,U] ) 




   

