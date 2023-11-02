#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold
import matplotlib.pyplot as mp
SIZE=np.array([1280, 720])

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    psr = f.psr 
    psr_names = f.psr_names

    print(psr)
    print(psr_names)

    title = "sblackbody_test.sh : %s " % f.base
    fig, ax = mp.subplots(figsize=SIZE/100.)
    fig.suptitle(title)
    for i in range(len(psr)):
        nm = psr[i,:,0]
        bb = psr[i,:,1]
        bb /= bb.sum()
        ax.plot( nm, bb, label=psr_names[i] )
    pass
    ax.legend()    
    fig.show()

    
