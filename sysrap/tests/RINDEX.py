#!/usr/bin/env python

import os, numpy as np
import matplotlib.pyplot as mp
SIZE=np.array([1280, 720])

if __name__ == '__main__':
    path = os.path.expandvars("$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/stree/material/$material/RINDEX.npy")
    a = np.load(path)
    print(path)
    print(a)

    ri = a[:,1] 
    eV = a[:,0]*1e6
    hc_eVnm = 1239.84198
    nm = hc_eVnm/eV    ## 330,331,...,599,600  very close to integer nm with 1239.84198


    fig, ax = mp.subplots(figsize=SIZE/100.)
    fig.suptitle(path)

    ax.plot( nm, ri )
    ax.scatter( nm, ri )

    ax.axvline(200, dashes=[2,1])
    ax.axvline(120, dashes=[1,1])
    ax.axvline(80,  dashes=[3,1])

    fig.show() 


pass

