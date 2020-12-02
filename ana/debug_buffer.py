#!/usr/bin/env python
"""
::

    run ~/opticks/ana/debug_buffer.py  

    In [10]: dgs
    Out[10]: 
    array([[-798499.8   ,   -4154.6196,       0.    ,       0.    ],
           [-804520.1   ,   -4182.02  ,       0.    ,       0.    ],
           [-804394.3   ,   -9223.    ,       0.    ,       0.    ],
           [-797320.    ,   -1448.3706,       0.    ,       0.    ],
           [-799486.75  ,   -1190.6932,       0.    ,       0.    ],
           [-798495.7   ,   -4151.6196,       0.    ,       0.    ],
           [-796048.1   ,   -1436.3706,       0.    ,       0.    ],



"""
import os, numpy as np
np.set_printoptions(suppress=True)

os.environ.setdefault("OPTICKS_EVENT_BASE",os.path.expandvars("/tmp/$USER/opticks"))
path = os.path.expandvars("$OPTICKS_EVENT_BASE/G4OKTest/evt/g4live/natural/1/dg.npy")
dg = np.load(path)

#sensorIndex = dg[:,0,3].view(np.uint32)
tid = dg[:,0,3].view(np.uint32)    

#sel = sensorIndex > 0
sel = tid > 0x5000000   # for DYB this means landing (but not necessarily "hitting") a volume of the instanced PMT assembly   

dgi = tid[sel]
dgs = dg[sel]

 



