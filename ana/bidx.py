#!/usr/bin/env python
"""
Hmm this duplicates ana/boundary_counts.py 
"""
import numpy as np
from opticks.ana.key import keydir
from opticks.ana.blib import BLib

KEYDIR=keydir()
blib = BLib()

if __name__ == '__main__':
    avi = np.load(os.path.join(KEYDIR, "GNodeLib/all_volume_identity.npy"))

    bidx = ( avi[:,2] >>  0)  & 0xffff 
    midx = ( avi[:,2] >> 16)  & 0xffff 

    b,n = np.unique( bidx, return_counts=True)

    for i in range(len(b)): 
        print("%3d : %7d : %s " % (b[i],n[i],blib.bname(b[i])))
    pass

    


