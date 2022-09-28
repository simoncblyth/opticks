#!/usr/bin/env python

import numpy as np

if __name__ == '__main__':
    a00 = np.load("/tmp/ntds0/000/U4Scintillation_Debug.npy").reshape(-1,8)
    a01 = np.load("/tmp/ntds0/001/U4Scintillation_Debug.npy").reshape(-1,8)
    a30 = np.load("/tmp/ntds3/000/U4Scintillation_Debug.npy").reshape(-1,8)
    a31 = np.load("/tmp/ntds3/001/U4Scintillation_Debug.npy").reshape(-1,8)

    print("a00 %s " % str(a00.shape))
    print("a01 %s " % str(a01.shape))
    print("a30 %s " % str(a30.shape))
    print("a31 %s " % str(a31.shape))




