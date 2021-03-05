#!/usr/bin/env python

import numpy as np
np.set_printoptions(linewidth=200, edgeitems=10) 

if __name__ == '__main__':

    a = np.load("/tmp/curand_skipahead_1.npy")
    b = np.load("/tmp/curand_skipahead_2.npy")

    assert np.all( a[:,:4] == b[:,:4]) 
    assert np.all( a[:,12:] == b[:,12:]) 


