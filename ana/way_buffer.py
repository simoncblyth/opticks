#!/usr/bin/env python
"""
::

    run ~/opticks/ana/way_buffer.py  

"""
import os, numpy as np
np.set_printoptions(suppress=True)

tst = "OKTest"

os.environ.setdefault("OPTICKS_EVENT_BASE",os.path.expandvars("/tmp/$USER/opticks"))
path = os.path.expandvars("$OPTICKS_EVENT_BASE/%s/evt/g4live/torch/1/wy.npy" % tst)
wy = np.load(path).reshape(-1,4)

 



