#!/usr/bin/env python

import os, numpy as np
import matplotlib.pyplot as mp
SIZE = np.array([1280, 720]) 

if __name__ == '__main__':
    a = np.load( os.path.join(os.environ["FOLD"], "morton_circle_demo.npy") )
    print(a)

    fig, ax = mp.subplots(figsize=SIZE/100.)
    ax.set_aspect('equal')

    ax.scatter( a[:,0], a[:,1], s=0.1 )
    fig.show()


