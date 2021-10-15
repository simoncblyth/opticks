#!/usr/bin/env python
"""

::

    ipython -i test/QEventTest.py 

"""

import os, numpy as np

if __name__ == '__main__':
    gs = np.load(os.path.expandvars("/tmp/$USER/opticks/qudarap/QEventTest/cegs.npy"))
    print(gs)
    print(gs.view(np.int32)) 

