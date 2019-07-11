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
from opticks.ana.profile import Profile 

if __name__ == '__main__':
    ok = opticks_main(doc=__doc__)  
    log.info(ok.brief)

    op = Profile(ok) 
    op.deltaVM()

    a = op.a  
    l = op.l 

    plt.plot( op.t, op.v, 'o' )
    plt.ion()
    plt.show()

    print(op)


