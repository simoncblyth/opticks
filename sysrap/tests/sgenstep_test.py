#!/usr/bin/env python
"""
sgenstep_test.py
================

~/o/sysrap/tests/sgenstep_test.sh 

"""
import os 
import numpy as np
u_ = lambda a:np.c_[np.unique(a,return_counts=True)]
MODE = int(os.environ.get("MODE","0"))

pvplt_plotter = None
pvplt_arrows = None
pvplt_lines = None

if MODE in [2,3]:
    try:
        from opticks.ana.pvplt import pvplt_plotter, pvplt_arrows, pvplt_lines
    except ImportError:
        pass
    pass
pass 
    

if __name__ == '__main__':
    path = os.path.expandvars("$GSPATH")
    gs = np.load(path)
    print(path)
    print(gs.shape)

    gentype = gs[:,0,0].view(np.int32)
    trackid = gs[:,0,1].view(np.int32)
    matline = gs[:,0,2].view(np.int32)
    numphoton = gs[:,0,3].view(np.int32)

    print("gentype\n", gentype)
    print("trackid\n", trackid)
    print("matline\n", matline)
    print("numphoton\n", numphoton)

    u_gentype = u_(gentype)
    u_trackid = u_(trackid)
    u_matline = u_(matline)
    u_numphoton = u_(numphoton)

    print("u_gentype\n", u_gentype)
    print("u_trackid\n", u_trackid)
    print("u_matline\n", u_matline)
    print("u_numphoton\n", u_numphoton)

    pos = gs[:,1,:3]
    print("pos\n", pos) 

    dp = gs[:,2,:3] 
    print("dp:DeltaPosition\n", dp)
 
    label = "sgenstep_test %s" % path
    print("label\n", label)

    if MODE == 3:
        pl = pvplt_plotter(label=label) 
        pvplt_viewpoint( pl )
        pvplt_lines( pl, pos, dp )
        cp = pl.show()
        print("cp\n", cp)
    else:
        print("MODE:%d not plotting : use MODE=3 to pvplt " % MODE ) 
    pass
