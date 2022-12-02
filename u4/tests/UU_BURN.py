#!/usr/bin/env python

import os, numpy as np
UU_BURN = os.environ.get("UU_BURN", "/tmp/UU_BURN.npy" )

if __name__ == '__main__':

    a = np.array( [[10,3]], dtype=np.int32 ) 

    print(" UU_BURN : %s  a : %s " % ( UU_BURN, str(a.shape)) )
    print(a)

    np.save(UU_BURN, a )



