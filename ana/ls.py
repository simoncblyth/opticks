#!/usr/bin/env python

import numpy as np

np.set_printoptions(suppress=True)

from opticks.ana.material import Material

if __name__ == '__main__':
    mat = Material("LS")
    wl = np.linspace(60.,820., 39)
    tab = mat.table(wl).reshape(-1,6)

    #print(mat.hdr())
    #print(tab)
  
    qwn = "wavelen rindex abslen scatlen reemprob groupvel".split()
    print("".join(list(map(lambda _:" %10s " % _, qwn))))
 
    fmt = " %10.3f " * 6
    for row in tab:
        print( fmt % tuple(row) )
    pass
    #print(tab.shape)
pass

