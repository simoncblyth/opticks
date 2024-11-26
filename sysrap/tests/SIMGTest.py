#!/usr/bin/env python
"""
::

    IMGPATH=$HOME/rocket.jpg ~/o/sysrap/tests/SIMGTest.sh ana


"""
import os, numpy as np 
import matplotlib.pyplot as plt


if __name__ == '__main__':
    a = np.load(os.path.expandvars("$NPYPATH"))
    print("a.shape : %s " % repr(a.shape))

    fig, axs = plt.subplots(2,2)

    ax = axs[0,0]
    ax.imshow(a)
    ax.set_xlabel("a")

    ax = axs[0,1]
    ax.imshow(a[:,:,0], label="R")
    ax.set_xlabel("a[:,:,0]")

    ax = axs[1,0]
    ax.imshow(a[:,:,1])
    ax.set_xlabel("a[:,:,1]")

    ax = axs[1,1]
    ax.imshow(a[:,:,2])
    ax.set_xlabel("a[:,:,2]")


    plt.show()

