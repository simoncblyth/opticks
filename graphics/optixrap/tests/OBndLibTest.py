#!/usr/bin/env python
import numpy as np
np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True, precision=3)

if __name__ == '__main__':
    bb = np.load("/tmp/OBndLib_convert_bndbuf.npy")
    print bb.shape
    print bb[13,0]


