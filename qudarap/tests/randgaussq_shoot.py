#!/usr/bin/env python

import numpy as np 
SIZE = np.array([1280, 720])
import matplotlib.pyplot as mp


if __name__ == '__main__':
    a = np.load(os.path.expandvars("$FOLD/RandGaussQ_shoot.npy"))
    fig, ax = mp.subplots(figsize=SIZE/100.)
    ax.hist(a, bins=100)
    fig.show() 



