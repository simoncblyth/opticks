#!/usr/bin/env python

import os, numpy as np
from opticks.ana.base import opticks_main

if __name__ == '__main__':
    args = opticks_main()
    a = np.load(os.path.expandvars("$TMP/OOMinimalTest.npy"))
    print a



