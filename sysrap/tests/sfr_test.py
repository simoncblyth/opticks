#!/usr/bin/env python

import os, numpy as np

if __name__ == '__main__':
    path = os.path.expandvars("$FOLD/sfr.npy")
    a = np.load(path)
    print(a)
    print(a[0].view(np.int64))
pass
