#!/usr/bin/env python

import numpy as np
SIZE = np.array([1280, 720])
import matplotlib.pyplot as mp


def test_transformQuick():
    a = np.load(os.path.expandvars("$FOLD/test_transformQuick.npy"))
    fig, ax = mp.subplots(figsize=SIZE/100.)
    ax.scatter( a[:,0], a[:,1], s=0.1 )
    fig.show() 

def test_shoot():
    a = np.load(os.path.expandvars("$FOLD/test_shoot.npy"))
    fig, ax = mp.subplots(figsize=SIZE/100.)
    ax.hist(a, bins=100)
    fig.show() 

if __name__ == '__main__':
    test_shoot()

pass
