#!/usr/bin/env python
import numpy as np

if __name__ == '__main__':
    e = np.load("/tmp/e.npy")

    ix = np.argsort(e)   

    i0 = np.load("/tmp/i0.npy")
    i1 = np.load("/tmp/i1.npy")
    i2 = np.load("/tmp/i2.npy")

    i0_match = np.all( ix == i0 )
    i1_match = np.all( ix == i1 )
    i2_match = np.all( ix == i2 )

    print( "i0_match %d " % i0_match )
    print( "i1_match %d " % i1_match )
    print( "i2_match %d " % i2_match )

pass

