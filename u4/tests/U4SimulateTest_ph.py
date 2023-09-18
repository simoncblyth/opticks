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
N =  int(os.environ.get("VERSION", "-1"))
VERSION = N 
CMDLINE = "N=%d ./U4SimulateTest.sh ph" % N

MODE =  int(os.environ.get("MODE", "2"))
assert MODE in [0,2,3]
PID = int(os.environ.get("PID", -1))
if PID == -1: PID = int(os.environ.get("OPTICKS_G4STATE_RERUN", -1))

if MODE > 0:
    from opticks.ana.pvplt import * 
pass

if __name__ == '__main__':

    t = Fold.Load(symbol="t")
    tp = Fold.Load(symbol="tp", parent=True)

    if not tp is None:
        rr = np.array([tp.run_meta.T_BeginOfRun, tp.run_meta.T_EndOfRun], dtype=np.uint64 ) 
        rre_ = "np.c_[rr.view(\"datetime64[us]\")]" 
        print(rre_)
        print(eval(rre_))
    pass

    print(repr(t))
    print( "MODE:%d" % (MODE) )
    print( "N:%d" % (N) )

    if 'SPECS' in os.environ:
        SPECS = np.array(t.U4R_names.lines)
        st_ = t.aux[:,:,2,3].view(np.int32)
        st = SPECS[st_]   # get out of bounds, when using st_ for other checks
    else:
        SPEC, st_, st = None, None, None
    pass


    axes = 0, 2  # X,Z
    H,V = axes 
    label = LABEL 

    #pos = t.photon[:,0,:3]
    pos = t.record[:,:,0,:3].reshape(-1,3)   # xyz : all photons, all steps
    q_ = t.seq[:,0]    #  t.seq shape eg (1000, 2, 2)  
    q = ht.seqhis(q_)    # history label eg b'TO BT BT SA ... lots of blankspace...'  
    qq = ht.Convert(q_)  # (n,32) int8 : for easy access to nibbles 
    n = np.sum( seqnib_(q_), axis=1 ) 


    ReplicaNumber=False
    if ReplicaNumber:
        ## ReplicaNumber : but when not more than one of each type of volume this is -1
        rp = t.record[...,1,3].view(np.int32) 
        np.set_printoptions(edgeitems=50)  

        u_rp, i_rp, v_rp, n_rp = np.unique(rp, axis=0, return_index=True, return_inverse=True, return_counts=True ) 
        print("\nnp.c_[np.arange(len(i_rp)),i_rp,n_rp,u_rp] ## unique ReplicaNumber sequences ")
        print(np.c_[np.arange(len(i_rp)),i_rp,n_rp,u_rp])
        print("\nlen(v_rp) : %d ## v_rp : unique array indices that reproduce original array  " % len(v_rp))
        assert len(rp) == len(v_rp) 
    pass

    ## uniqing the "|S96" label because NumPy lacks uint128  
    qu, qi, qn = np.unique(q, return_index=True, return_counts=True)  
    quo = np.argsort(qn)[::-1]  
    expr_ = "np.c_[qn,qi,qu][quo]"
    expr = eval(expr_)
  
    
    print("\n%s  ## unique histories qu in descending count qn order, qi first index " % expr_ )
    print(expr) 

    numpystr_ = lambda _:"%5s" % _.strip().decode("utf-8")
    int_ = lambda _:"%5d" % _
    formatter = {'numpystr':numpystr_, 'int':int_}   
    ## HMM: when concatenating all the types get converted to numpystr, so not much control 
    # idea for more control of array presentation
    print("\n",np.array2string(expr,formatter=formatter))



    ws_ = 0 
    ws = np.where( q[:,0] == qu[quo][ws_] )   # select photons with the ws_ most prolific history    


    #print("\nq[v_rp == 0]  ## history flag sequence for unique ReplicaNumber sequence 0"  )
    #print(repr(q[v_rp == 0]))

    print("\nnp.unique(n, return_counts=True) ## occupied nibbles  ")
    print(repr(np.unique(n, return_counts=True)))
    
    print("\nq[n > 16]  ## flag sequence of big bouncers  ")
    print(repr(q[n>16]))  

    n_cut = 10

    expr = "q[n > %d]" % n_cut 
    print("\n%s  ## flag sequence of big bouncers  " % expr )
    print(repr(eval(expr)))  

    expr = "np.c_[n,q][n>%d]" % (n_cut)
    print("\n%s  ## nibble count with flag sequence of big bouncers  " % expr )
    print(eval(expr))  

    print("\nnp.where(n > 28)  ## find index of big bouncer " )
    print(np.where(n > 28)) 

    expr = "np.c_[np.where(n>%d)[0],q[n > %d]]" % (n_cut,n_cut)
    print("\n%s  ## show indices of multiple big bouncers together with history " % expr)
    print(eval(expr))

    expr = " t.record[%d,:n[%d],0] " % (PID,PID)
    print("\n%s  ## show step record points for PID %d  " % (expr, PID))
    print(eval(expr))

    expr = " np.where( t.record[:500,:,0,2] < 0 ) "
    print("\n%s ## look for records with -Z positions in the first half, that all start in +Z " % expr )
    print(eval(expr))

    expr = " np.where( t.record[500:,:,0,2] > 0 ) "
    print("\n%s ## look for records with +Z positions in the second half, that all start in -Z " % expr )
    print(eval(expr))

    print("\nt.base : %s  VERSION: %d " % (t.base, VERSION))


    pos = t.photon[:,0,:3]  
    flagmask = t.photon[:,3,3].view(np.int32) 

    expr = " np.where( np.logical_and( pos[:,0] < -150, pos[:,2] > 200 )  ) "
    print("\n%s ## common break out line, use t.photon end position to get indices " % expr )
    print(eval(expr))
  
    expr = " q[np.where( np.logical_and( pos[:,0] < -150, pos[:,2] > 200 ))][:3] "
    print("\n%s ## common break out line, use t.photon end position to get indices wline and see what the q histories are " % expr )
    print(eval(expr))


    SURFACE_DETECT = 0x1 << 6 
    sd = flagmask & SURFACE_DETECT
    w_sd = np.where( sd )[0]


    x_midline = np.logical_and( pos[:,0] > -251, pos[:,0] < -249 )    
    z_midline = np.logical_and( pos[:,2] > -250, pos[:,2] <  250 )    
    xz_midline = np.logical_and( x_midline, z_midline )
    w_midline = np.where(xz_midline)[0]  


    yy = t.record[:,:,0,1]
    myy = np.max( np.abs(yy), axis=1 )     ## max absolute y of all step points for each photon

    wyy0 = np.where( myy == 0. )[0]        ## photon indices that always stay in the plane
    wyy1 = np.where( myy > 0 )[0]           ## photon indices with deviations out of plane 
    sd0 = np.logical_and( sd, myy == 0. )   ## mask SD photons staying in plane


    #ppos0_ = "None"
    #ppos0_ = "pos #   "
    ppos0_ = "t.record[:,0,0,:3] # 0-position   "

    #ppos1_ = "None"
    ppos1_ = "t.record[:,1,0,:3] # 1-position   "

    #ppos2_ = "None"
    ppos2_ = "t.record[:,2,0,:3] # 2-position   "


    #ppos3_ = "None"
    ppos3_ = "t.photon[:,0,:3]"

    ppos0  = eval(ppos0_)
    ppos1  = eval(ppos1_) 
    ppos2  = eval(ppos2_) 
    ppos3  = eval(ppos3_) 

    ppos = t.record[:,:,0,:3].reshape(-1,3)



    elem = []
    elem.append(CMDLINE)
    if not ppos0 is None: elem.append("b:%s" % ppos0_)
    if not ppos1 is None: elem.append("r:%s" % ppos1_)
    if not ppos2 is None: elem.append("g:%s" % ppos2_)
    if not ppos3 is None: elem.append("c:%s" % ppos3_)
    label = "\n".join(elem)


    if MODE == 0:
        print("not plotting as MODE 0  in environ")
    elif MODE == 2:
        fig, axs = mpplt_plotter(label=label)
        assert len(axs) == 1 
        ax = axs[0]      

        ax.set_ylim(-250,250)
        ax.set_xlim(-500,500)

        if not ppos0 is None: ax.scatter( ppos0[:,H], ppos0[:,V], s=1, c="b" )  
        if not ppos1 is None: ax.scatter( ppos1[:,H], ppos1[:,V], s=1, c="r" )  
        if not ppos2 is None: ax.scatter( ppos2[:,H], ppos2[:,V], s=1, c="g" )  
        if not ppos3 is None: ax.scatter( ppos3[:,H], ppos3[:,V], s=1, c="c" )  

        fig.show()
    elif MODE == 3:
        pl = pvplt_plotter(label)
        os.environ["EYE"] = "0,100,165"
        os.environ["LOOK"] = "0,0,165"
        pvplt_viewpoint(pl)
        pl.add_points(ppos )
        pl.show()
    pass
pass
