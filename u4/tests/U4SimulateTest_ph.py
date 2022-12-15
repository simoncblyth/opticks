#!/usr/bin/env python
"""
U4SimulateTest_ph.py
========================

::

    u4t
    ./U4SimulateTest.sh ph
    ./U4SimulateTest.sh nph

"""
import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.p import * 

LABEL = os.environ.get("LABEL", "U4SimulateTest_ph.py" )
VERSION =  int(os.environ.get("VERSION", "-1"))
MODE =  int(os.environ.get("MODE", "2"))
assert MODE in [0,2,3]
PID = int(os.environ.get("PID", -1))
if PID == -1: PID = int(os.environ.get("OPTICKS_G4STATE_RERUN", -1))

if MODE > 0:
    from opticks.ana.pvplt import * 
pass

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))
    print( " MODE : %d " % (MODE) )

    axes = X, Z 
    H,V = axes 
    label = LABEL 

    #pos = t.photon[:,0,:3]
    pos = t.record[:,:,0,:3].reshape(-1,3) 
    q_ = t.seq[:,0]
    q = ht.seqhis(q_)

    ## ReplicaNumber
    rp = t.record[...,1,3].view(np.int32) 
    np.set_printoptions(edgeitems=50)  

    u_rp, i_rp, v_rp, n_rp = np.unique(rp, axis=0, return_index=True, return_inverse=True, return_counts=True ) 
    print("\nnp.c_[np.arange(len(i_rp)),i_rp,n_rp,u_rp] ## unique ReplicaNumber sequences ")
    print(np.c_[np.arange(len(i_rp)),i_rp,n_rp,u_rp])
    print("\nlen(v_rp) : %d ## v_rp : unique array indices that reproduce original array  " % len(v_rp))
    assert len(rp) == len(v_rp) 


    qu, qi, qn = np.unique(q, return_index=True, return_counts=True)  
    quo = np.argsort(qn)[::-1]  
    expr = "np.c_[qn,qi,qu][quo]"
    
    print("\n%s  ## unique histories qu in descending count qn order, qi first index " % expr )
    print(eval(expr))  

    print("\nq[v_rp == 0]  ## history flag sequence for unique ReplicaNumber sequence 0"  )
    print(repr(q[v_rp == 0]))

    n = np.sum( seqnib_(q_), axis=1 ) 
    print("\nnp.unique(n, return_counts=True) ## occupied nibbles  ")
    print(repr(np.unique(n, return_counts=True)))
    
    print("\nq[n > 16]  ## flag sequence of big bouncers  ")
    print(repr(q[n>16]))  

    cut = 10

    expr = "q[n > %d]" % cut 
    print("\n%s  ## flag sequence of big bouncers  " % expr )
    print(repr(eval(expr)))  

    expr = "np.c_[n,q][n>%d]" % (cut)
    print("\n%s  ## nibble count with flag sequence of big bouncers  " % expr )
    print(eval(expr))  

    print("\nnp.where(n > 28)  ## find index of big bouncer " )
    print(np.where(n > 28)) 

    expr = "np.c_[np.where(n>%d)[0],q[n > %d]]" % (cut,cut)
    print("\n%s  ## show indices of multiple big bouncers together with history " % expr)
    print(eval(expr))

    expr = " t.record[%d,:n[%d],0] " % (PID,PID)
    print("\n%s  ## show step record points for PID %d  " % (expr, PID))
    print(eval(expr))

    expr = " np.where( t.record[:500,:,0,2] < 0 ) "
    print("\n%s ## look for records with -Z positions in the first half, that all start in +Z ")
    print(eval(expr))

    expr = " np.where( t.record[500:,:,0,2] > 0 ) "
    print("\n%s ## look for records with +Z positions in the second half, that all start in -Z ")
    print(eval(expr))

    print("\nt.base : %s  VERSION: %d " % (t.base, VERSION))


    if MODE == 0:
        print("not plotting as MODE 0  in environ")
    elif MODE == 2:
        fig, ax = mpplt_plotter(label)
        ax.scatter( pos[:,H], pos[:,V], s=1 )  
        fig.show()
    elif MODE == 3:
        pl = pvplt_plotter(label)
        os.environ["EYE"] = "0,100,165"
        os.environ["LOOK"] = "0,0,165"
        pvplt_viewpoint(pl)
        pl.add_points( pos )
        pl.show()
    pass
pass
