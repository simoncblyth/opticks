#!/usr/bin/env python

import os, numpy as np 
import matplotlib.pyplot as plt

if __name__ == '__main__':
    a = np.load(os.path.expandvars("$NPY"))
    print("a.shape : %s " % repr(a.shape))
    fig, ax = plt.subplots(1)

    ax.imshow(a)
    ax.set_xlabel("a")

    plt.show()


