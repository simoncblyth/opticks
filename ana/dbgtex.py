#!/usr/bin/env python
"""
::

    export OPTICKS_KEYDIR=/usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/c250d41454fba7cb19f3b83815b132c2/1
    ipython -i -- $(which dbgtex.py) 

"""
import os, numpy as np

if __name__ == '__main__':

    t = np.load(os.path.expandvars("$OPTICKS_KEYDIR/dbgtex/buf.npy"))
    print t 
    o = np.load(os.path.expandvars("$OPTICKS_KEYDIR/dbgtex/obuf.npy"))
    print o




