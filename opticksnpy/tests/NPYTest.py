#!/usr/bin/env python

import numpy as np

def test_repeat():
    # see tests/NPYTest.cc:test_repeat

    aa = np.load("/tmp/aa.npy")
    bb = np.load("/tmp/bb.npy")
    cc = np.load("/tmp/cc.npy")

    bbx = np.repeat(aa, 10, axis=0) ;
    assert np.all(bb == bbx)

    ccx = np.repeat(aa, 10, axis=0).reshape(-1,10,1,4)
    assert np.all(cc == ccx)



if __name__ == '__main__':
    test_repeat()



