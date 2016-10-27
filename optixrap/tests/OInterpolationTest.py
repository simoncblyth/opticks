#!/usr/bin/env python

import os, numpy as np
np.set_printoptions(precision=3, suppress=True)

if __name__ == '__main__':
    a = np.load(os.path.expandvars("$TMP/OInterpolationTest/out.npy")).reshape(-1,4,2,39,4) 
    b = np.load(os.path.expandvars("$IDPATH/GBndLib/GBndLib.npy"))
    assert np.all(a == b)




