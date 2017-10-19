#!/usr/bin/env python

TORCH = 4096 

import os, numpy as np
from opticks.ana.base import opticks_main

if __name__ == '__main__':
    args = opticks_main()
 
    a = np.load(os.path.expandvars("$TMP/trivialCheck.npy"))
    i = a.view(np.int32)

    ## these are specific to default TORCH genstep

    assert np.all(i[:,2,0] == TORCH)
    assert np.all(i[:,2,3] == 100000)   

    assert np.all( np.arange(len(i), dtype=np.int32) == i[:,3,0] )    # photon_id
    assert np.all( np.arange(len(i), dtype=np.int32)*4 == i[:,3,1] )  # photon_offset

    assert np.all(i[:,3,2] == 0)           # genstep_id  (all zero as only 1 genstep for default TORCH)
    assert np.all(i[:,3,3] == i[:,3,2]*6)  # genstep_offset




