#!/usr/bin/env python

import os, numpy as np

if __name__ == '__main__':
    p0 = os.path.expandvars("$FOLD/000/U4Hit_Debug.npy")
    a0 = np.load(p0)
    print( " p0 %s a0 %s " % (p0, str(a0.shape)))
