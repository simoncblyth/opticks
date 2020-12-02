#!/usr/bin/env python
"""
::

    run ~/opticks/ana/debug_buffer.py  

"""
import os, numpy as np
np.set_printoptions(suppress=True)

os.environ.setdefault("OPTICKS_EVENT_BASE",os.path.expandvars("/tmp/$USER/opticks"))
path = os.path.expandvars("$OPTICKS_EVENT_BASE/G4OKTest/evt/g4live/natural/1/dg.npy")
dg = np.load(path)

sensorIndex = dg[:,0,3].view(np.uint32)
#tid = dg[:,0,3].view(np.uint32)    

sel = sensorIndex > 0
#sel = tid > 0x5000000   # for DYB this means landing (but not necessarily "hitting") a volume of the instanced PMT assembly   

dgi = sensorIndex[sel]
dgs = dg[sel]

 



