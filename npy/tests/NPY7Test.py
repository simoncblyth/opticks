#!/usr/bin/env python 
"""

::

    ipython -i NPY7Test.py 


"""
import matplotlib.pyplot as plt 
import os, numpy as np

FOLD = "/tmp/NPY7Test/test_interp"

if __name__ == '__main__':
    src = np.load(os.path.join(FOLD,"src.npy"))
    dst = np.load(os.path.join(FOLD,"dst.npy"))

    fig, ax = plt.subplots(figsize=[12.8,7.2])
    ax.scatter( src[:,0], src[:,1], label="src" )
    ax.plot( dst[:,0], dst[:,1], label="dst" )
    ax.legend()

    fig.show()
    plt.ion()



