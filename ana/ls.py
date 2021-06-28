#!/usr/bin/env python
"""
::

    ipython --pdb -i ls.py 

"""
import numpy as np

np.set_printoptions(suppress=True)

from opticks.ana.proplib import PropLib
from opticks.ana.material import Material
import matplotlib.pyplot as plt


if __name__ == '__main__':

    matname = "LS"
    #matname = "Pyrex"
    #matname = "Air"
    #matname = "Water"
    mat = Material(matname)

    #wl = PropLib.COARSE_DOMAIN
    wl = PropLib.FINE_DOMAIN

    tab = mat.table(wl).reshape(-1,6)

    #print(mat.hdr())
    #print(tab)
 
    #      0       1      2      3       4        5 
    qwn = "wavelen rindex abslen scatlen reemprob groupvel".split()
    print("".join(list(map(lambda _:" %10s " % _, qwn))))
 
    fmt = " %10.3f " * 6
    for row in tab:
        print( fmt % tuple(row) )
    pass
    #print(tab.shape)

    fig, axs = plt.subplots(2,1)  
    fig.suptitle(matname)

    ax = axs[0]
    ax.plot( tab[:,0], tab[:,1], label=qwn[1] )
    ax.plot( tab[:,0], tab[:,4], label=qwn[4] )
    ax.plot( tab[:,0], tab[:,5]/300., label="%s/300" % qwn[5] )
    ax.legend()

    ax = axs[1]
    ax.plot( tab[:,0], tab[:,2], label=qwn[2] )
    ax.plot( tab[:,0], tab[:,3], label=qwn[3] )
    ax.legend()

    fig.show()

