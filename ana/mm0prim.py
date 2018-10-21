#!/usr/bin/env python
"""
Hmm need to make connection to the volume traversal index 
"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.geocache import keydir
from opticks.ana.prim import Dir

import matplotlib.pyplot as plt


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    ok = os.environ["OPTICKS_KEY"]
    kd = keydir(ok)

    names = False
    if names:
        pvn = np.loadtxt(os.path.join(kd, "GNodeLib/PVNames.txt" ), dtype="|S100" )
        lvn = np.loadtxt(os.path.join(kd, "GNodeLib/LVNames.txt" ), dtype="|S100" )
    else:
        pvn = None
        lvn = None
    pass

    ce = np.load(os.path.join(kd, "GMergedMesh/0/center_extent.npy"))


    log.info(kd)
    assert os.path.exists(kd), kd 
    os.environ["IDPATH"] = kd 

    d = Dir(os.path.expandvars("$IDPATH/GParts/0"))   ## mm0 analytic
    #print d     
    pp = d.prims

    sli = slice(0,None)

    prims = []
    for p in pp[sli]:
        if p.lvIdx in [8,9]: continue   # too many  
        if p.numParts > 1: continue    # skip compounds for now
        prims.append(p)
        #print(repr(p)) 
        #print(str(p)) 
        if names:
            vol = p.idx[0]
            pv = pvn[vol]
            lv = lvn[vol]
            print(pv)
            print(lv)
        pass
    pass

    plt.ion()
    fig = plt.figure(figsize=(6,5.5))
    plt.title("mm0prim")
    ax = fig.add_subplot(111)
    #sz = 150 
    sz = 50 
    ax.set_ylim([-sz,sz])
    ax.set_xlim([-sz,sz])

    sc = 1000

    for i,p in enumerate(prims):
        assert len(p.parts) == 1 
        pt = p.parts[0]
        print(repr(p)) 
        #print(pt) 
        #print(pt.tran) 
        sh = pt.as_shape("prim%s" % i, sc=sc ) 
        if sh is None: 
           print(str(p))
           continue
        #print(sh)
        for pa in sh.patches():
            ax.add_patch(pa)
        pass
    pass

    dtype = np.float32
    n = 3 
    eye = np.zeros( (n, 3), dtype=dtype )
    look = np.zeros( (n, 3), dtype=dtype )
    up = np.zeros( (n, 3), dtype=dtype )

    eye[0] = [-2, 0, -1] 
    eye[1] = [ 0, 0, 0.5]
    eye[2] = [ 0, 0, 2]


    look[:-1] = eye[1:]
    look[-1] = eye[0]
    up[:] = [0,0,1] 

    v = np.zeros( (n,4,4), dtype=np.float32)
    v[:,0,:3] = eye 
    v[:,1,:3] = look
    v[:,2,:3] = up

    sc = 30.
    ax.plot( sc*eye[:,0], sc*eye[:,2]  )

    np.save("/tmp/flightpath.npy", v ) 

    fig.show()


