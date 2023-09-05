#!/usr/bin/env python
"""


"""
import os, numpy as np
import matplotlib.pyplot as mp  
SIZE = np.array([1280, 720])

np.set_printoptions(precision=5)

if __name__ == '__main__':
    path = os.path.expandvars("$FOLD/test_erfcinvf.npy")
    print(path)
    a = np.load(path)
    print(a)


    fig, ax = mp.subplots(figsize=SIZE/100.)


    ref = "/tmp/S4MTRandGaussQTest/a.npy"
    if True and os.path.exists(ref):
        b = np.load(ref) 
        ax.scatter( b[:,0], b[:,1], s=0.1 )
    pass

    ax.scatter( a[:,0], a[:,2], s=0.1 )


    ab = b[:,1] - a[:,2]


    fig.show()




