#!/usr/bin/env python

import os, numpy as np


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3 )
    f = np.load(os.path.expandvars("$TMP/evt/dayabay/torch/1/gs.npy"))
    i = f.view(np.int32)
    print i 

