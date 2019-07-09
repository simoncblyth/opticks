#!/usr/bin/env python
"""

::

    ip tprofile.py 

::

    ip () 
    { 
        local py=${1:-dummy.py};
        shift;
        ipython --pdb $(which $py) -i $*
    }


"""
from __future__ import print_function
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from opticks.ana.main import opticks_main
from opticks.ana.nload import np_load
from opticks.ana.nload import tagdir_, stamp_


if __name__ == '__main__':
    ok = opticks_main(doc=__doc__)  
    log.info(ok.brief)

    tag = None
    tfold = tagdir_( ok.det, ok.src, tag, pfx=ok.pfx )  
    path = os.path.join( tfold, "Opticks.npy")   # quads from OpticksProfile  
    print("path:%s stamp:%s " % (path, stamp_(path) ))

    a = np.load(path)

    t,dt,v,dv = a[:,0,0], a[:,0,1], a[:,0,2], a[:,0,3]
    plt.plot( t, v, 'o' )
    plt.show()

    plt.ion()



