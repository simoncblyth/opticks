#!/usr/bin/env python 

import numpy as np
from opticks.ana.key import keydir

if __name__ == '__main__':
    kd = keydir()
    path = os.path.join(kd, "GNodeLib/all_volume_LVNames.txt")

    lvn = np.loadtxt(path, dtype=np.object)
    print(" lvn.shape %s  path %s " % (str(lvn.shape), path) )

    u_lvn, u_lvn_counts = np.unique(lvn, return_counts=True)  

    for i in range(len(u_lvn)):
        print(" %5d : %6d : %50s " % (i, u_lvn_counts[i], u_lvn[i] ))
    pass

    




