#!/usr/bin/env python

import numpy as np

if __name__ == '__main__':
    NPY = os.environ.get("NPY","/tmp/stran_FromPair_checkIsIdentity_FAIL.npy" ) 
    a = np.load(NPY)
    symbol = "a"
    print(" %2s : %10s : %s " % ( symbol, str(a.shape), NPY )) 
