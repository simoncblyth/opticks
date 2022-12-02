#!/usr/bin/env python

import os, numpy as np

if __name__ == '__main__':
    a = np.load(os.path.expandvars("$FOLD/A_SPhoton_Debug.npy"))
    b = np.load(os.path.expandvars("$FOLD/B_SPhoton_Debug.npy"))
    print(a)
    print(b)
