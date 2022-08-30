#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
import matplotlib.pyplot as mp

SIZE = np.array([1280, 720]) 

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))

    t_pos = t.gs[:,5,:3]       

    fig, ax = mp.subplots(figsize=SIZE/100.)
    ax.set_aspect('equal')
    ax.scatter( t_pos[:,0], t_pos[:,2], label="t_pos", s=1 ) 

    fig.show()


