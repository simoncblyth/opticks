#!/usr/bin/env python
"""
::

    OKTest --save

    OEvent=INFO OKTest --save --compute --dumphit --dumphiy

    OEvent=INFO OKTest --save --dumphit --dumphiy    ##

    OEvent=INFO lldb_ -- OKTest --save --dumphit --dumphiy

    ipython -i ~/opticks/ana/way_buffer.py 

    ipython > run ~/opticks/ana/way_buffer.py  

::

    ox_path : /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/ox.npy ox.shape (10000, 4, 4) 
    wy_path : /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/wy.npy wy.shape (10000, 2, 4) 
    ht_path : /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/ht.npy ht.shape (87, 4, 4) 
    hy_path : /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/hy.npy hy.shape (87, 2, 4) 


"""
import os, numpy as np
np.set_printoptions(suppress=True)

tst = "OKTest"

os.environ.setdefault("OPTICKS_EVENT_BASE",os.path.expandvars("/tmp/$USER/opticks"))
path_ = lambda _:os.path.expandvars("$OPTICKS_EVENT_BASE/%s/evt/g4live/torch/1/%s.npy" % (tst,_))

wy_path = path_("wy")
ox_path = path_("ox")
ht_path = path_("ht")
hy_path = path_("hy")

wy = np.load(wy_path)
ox = np.load(ox_path)
ht = np.load(ht_path)
hy = np.load(hy_path)

print("ox_path : %s ox.shape %r " % (ox_path,ox.shape) )
print("wy_path : %s wy.shape %r " % (wy_path,wy.shape) )
print("ht_path : %s ht.shape %r " % (ht_path,ht.shape) )
print("hy_path : %s hy.shape %r " % (hy_path,hy.shape) )



