#!/usr/bin/env python
import numpy as np

if __name__ == '__main__':
    e = np.load("/tmp/e.npy")
    i = np.load("/tmp/i.npy")

    i2 = np.argsort(e)   
    assert np.all( i == i2 )
pass

