#!/usr/bin/env python

import os, numpy as np 
SIZE = np.array([1280, 720])
MODE = int(os.environ.get("MODE","0"))

if MODE == 2:
    import matplotlib.pyplot as mp
pass

if __name__ == '__main__':
    a = np.load(os.path.expandvars("$FOLD/RandGaussQ_shoot.npy"))
    print("a.shape\n",a.shape)

    if MODE == 2:
        fig, ax = mp.subplots(figsize=SIZE/100.)
        ax.hist(a, bins=100)
        fig.show() 
    pass
pass


