#!/usr/bin/env python

import numpy as np
SIZE = np.array([1280, 720])
import matplotlib.pyplot as mp
from opticks.ana.fold import Fold

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))
    a = t.a

    fig, ax = mp.subplots(figsize=SIZE/100.)
    
    ax.scatter( a[:,0], a[:,1], s=0.1 )

    fig.show() 


pass
