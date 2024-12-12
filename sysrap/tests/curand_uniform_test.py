#!/usr/bin/env python
"""


"""
import os, numpy as np
from opticks.ana.fold import Fold

#import matplotlib.pyplot as mp  
#SIZE = np.array([1280, 720])

np.set_printoptions(precision=5, linewidth=200, suppress=True )


if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    #a = np.load(os.path.expandvars("$FOLD/curand_uniform_test.npy"))
    #print("a.shape\n",a.shape)
    #print("a[:10]\n",a[:10])
    #print("a[-10:]\n",a[-10:])

