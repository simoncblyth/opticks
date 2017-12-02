#!/usr/bin/env python

import os, numpy as np

np.set_printoptions(precision=8, suppress=True)

if __name__ == '__main__':
    a = np.load(os.path.expandvars("$TMP/TRngBufTest.npy"))
    print a.shape
    print a
    print a.shape
    print a.dtype



