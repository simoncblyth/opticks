#!/usr/bin/env python

import os, numpy as np
np.set_printoptions(linewidth=200)
#np.set_printoptions(suppress=True, precision=3)
from opticks.ana.base import opticks_main

if __name__ == '__main__':
    args = opticks_main()
    bb = np.load("$TMP/OBndLib_convert_bndbuf.npy")
    print bb.shape
    print bb[13,0]


