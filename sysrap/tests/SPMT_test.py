#!/usr/bin/env python
"""
SPMT_test.py
==============

"""
import os, numpy as np
from opticks.ana.fold import Fold
import matplotlib.pyplot as plt
SIZE = np.array([1280,720])

if __name__ == '__main__':
    s = Fold.Load("$SFOLD", symbol="s")
    print(repr(s))

    f = s.get_ARTE
    print("f:s.get_ARTE") 
    print(repr(f))

    args = f.args.squeeze()
    spec = f.spec.squeeze()
    ss = f.ss.squeeze()
    ARTE = f.ARTE.squeeze()

    stack = f.stack.squeeze()
    ll = f.ll.squeeze() 
    comp = f.comp.squeeze() 
    art = f.art.squeeze() 

    nstack = f.nstack.squeeze()
    nll = f.nll.squeeze() 
    ncomp = f.ncomp.squeeze() 
    nart = f.nart.squeeze() 

    #_art = nart  # corresponds to just the -1 point in the mct scan  
    _art = art


    Rs = _art[...,0,0]
    Rp = _art[...,0,1]
    Ts = _art[...,0,2]
    Tp = _art[...,0,3]

    As = _art[...,1,0]
    Ap = _art[...,1,1]
    R_ = _art[...,1,2]
    T_ = _art[...,1,3]
      
    A_ = _art[...,2,0]
    A_R_T = _art[...,2,1]
    wl    = _art[...,2,2] 
    mct   = _art[...,2,3]

    xx  = _art[...,3,0]
    yy  = _art[...,3,1]
    zz  = _art[...,3,2]
    ww  = _art[...,3,3]

    pmtid = 0 

    opt = os.environ.get("OPT", "A_,R_,T_,As,Rs,Ts,Ap,Rp,Tp")
    title = "SPMT_test pmtid %d OPT %s " % (pmtid, opt) 
    fig, ax = plt.subplots(1, figsize=SIZE/100.)
    fig.suptitle(title)


    if "A_" in opt:ax.plot(  mct, A_, label="A_" )
    if "R_" in opt:ax.plot(  mct, R_, label="R_" )
    if "T_" in opt:ax.plot(  mct, T_, label="T_" )
    if "A_R_T" in opt:ax.plot( mct, A_R_T, label="A_R_T") 

    if "As" in opt:ax.plot(  mct, As, label="As" )
    if "Rs" in opt:ax.plot(  mct, Rs, label="Rs" )
    if "Ts" in opt:ax.plot(  mct, Ts, label="Ts" )

    if "Ap" in opt:ax.plot(  mct, Ap, label="Ap" )
    if "Rp" in opt:ax.plot(  mct, Rp, label="Rp" )
    if "Tp" in opt:ax.plot(  mct, Tp, label="Tp" )

    if "xx" in opt:ax.plot(  mct, xx, label="xx" )
    if "yy" in opt:ax.plot(  mct, yy, label="yy" )
    if "zz" in opt:ax.plot(  mct, zz, label="zz" )
    if "ww" in opt:ax.plot(  mct, ww, label="ww" )



    ax.legend()
    fig.show()


if 0:
    j = Fold.Load("$JFOLD", symbol="j")
    print(repr(j))

    #a = s.rindex
    #b = j.jpmt_rindex

    # compare stackspec between JPMT and SPMT 
    a = s.test.get_stackspec
    b = j.test.get_stackspec
    ab = np.abs(a-b)   
    print(" ab.max %s " % ab.max() )


    #expr = "a.reshape(-1,2)[:,0]"
    #print(expr) 
    #print(eval(expr)) 


    qi = s.test.get_pmtid_qe
    qc = s.test.get_pmtcat_qe
    ct = s.test.get_pmtcat 

    fig, ax = plt.subplots(1, figsize=[12.8, 7.2] )

    for i in range(3):
        ax.plot( qc[i,:,0], qc[i,:,1], label="qc[%d]"%i ) 
    pass
    ax.legend()    

    #for pmtid in range(25): 
    #    ax.plot( qi[pmtid,:,0], qi[pmtid,:,1], label=pmtid ) 
    #pass

    fig.show()

pass

