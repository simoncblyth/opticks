#!/usr/bin/env python
"""
SPMT_test.py
==============

"""
import os, numpy as np
from opticks.ana.fold import Fold
import matplotlib.pyplot as plt
SIZE = np.array([1280,720])
np.set_printoptions(edgeitems=16) 


def compare_stack_vs_stackNormal(f):
    """
    """
    print("\ncompare_stack_vs_stackNormal f.base : %s " % f.base)

    if getattr(f,"nart", None) == None:
        print("NO nart SKIP")
        return 
    pass 

    art = f.art.squeeze() 
    nart = f.nart.squeeze() 

    comp = f.comp.squeeze() 
    ncomp = f.ncomp.squeeze() 

    ll = f.ll.squeeze() 
    nll = f.nll.squeeze() 


    # 1. check ncomp is repeated ncomp[0] as stackNormal fixes mct at -1.f
    fab_ncomp = np.zeros_like(ncomp)  
    fab_ncomp[:] = ncomp[0]    
    assert np.all( fab_ncomp == ncomp )

    assert np.all( ncomp[:] == ncomp[0] ) # IS THIS TOTALLY SAME AS ABOVE ?


    # 2. check comp[0] == ncomp[0] as comp scan starts at mct -1
    assert np.all( comp[0] == ncomp[0] ) 

    fab_nart = np.zeros_like(nart)
    fab_nart[:] = nart[0]
    assert np.all( fab_nart == nart )

    assert np.all( nart[:] == nart[0] )  # IS THIS TOTALLY SAME AS ABOVE ?
    assert np.all( nll[:] == nll[0] )  # IS THIS TOTALLY SAME AS ABOVE ?





def check_nan(f):
    print("\ncheck_nan f.base %s " % f.base)
    qwns = "args spec extra ARTE stack ll comp art nstack nll ncomp nart".split()
    for qwn in qwns:
        q = getattr(f, qwn, None)
        if q is None: continue
        expr = "np.where( np.isnan(f.%s.ravel()) )[0]" % qwn  
        nan = eval(expr)
        print(" %-50s : %s " % (expr,eval(expr)))
    pass



if __name__ == '__main__':
    s = Fold.Load("$SFOLD/spmt", symbol="s")
    print(repr(s))

    f = Fold.Load("$SFOLD/sscan", symbol="f")
    print(repr(f))

    compare_stack_vs_stackNormal(f)
    check_nan(f)


    args = f.args.squeeze()
    spec = f.spec.squeeze()
    extra = f.extra.squeeze()
    ARTE = f.ARTE.squeeze()

    stack = f.stack.squeeze()
    ll = f.ll.squeeze() 
    comp = f.comp.squeeze() 
    art = f.art.squeeze() 


    As   = art[...,0,0]
    Ap   = art[...,0,1]
    Aa   = art[...,0,2]
    A_   = art[...,0,3]

    Rs   = art[...,1,0]
    Rp   = art[...,1,1]
    Ra   = art[...,1,2]
    R_   = art[...,1,3]

    Ts   = art[...,2,0]
    Tp   = art[...,2,1]
    Ta   = art[...,2,2]
    T_   = art[...,2,3]

    SF     = art[...,3,0]
    wl     = art[...,3,1] 
    ARTa   = art[...,3,2]
    mct    = art[...,3,3]


    pmtid = 0 

    opt = os.environ.get("OPT", "A_,R_,T_,As,Rs,Ts,Ap,Rp,Tp,Aa,Ra,Ta")
    title = "%s : pmtid %d OPT %s " % (s.base, pmtid, opt) 
    fig, ax = plt.subplots(1, figsize=SIZE/100.)
    fig.suptitle(title)

    if "As" in opt:ax.plot(  mct, As, label="As" )
    if "Ap" in opt:ax.plot(  mct, Ap, label="Ap" )
    if "Aa" in opt:ax.plot(  mct, Aa, label="Aa" )
    if "A_" in opt:ax.plot(  mct, A_, label="A_" )

    if "Rs" in opt:ax.plot(  mct, Rs, label="Rs" )
    if "Rp" in opt:ax.plot(  mct, Rp, label="Rp" )
    if "Ra" in opt:ax.plot(  mct, Ra, label="Ra" )
    if "R_" in opt:ax.plot(  mct, R_, label="R_" )

    if "Ts" in opt:ax.plot(  mct, Ts, label="Ts" )
    if "Tp" in opt:ax.plot(  mct, Tp, label="Tp" )
    if "Ta" in opt:ax.plot(  mct, Ta, label="Ta" )
    if "T_" in opt:ax.plot(  mct, T_, label="T_" )

    if "SF" in opt:ax.plot(  mct, SF, label="SF") 
    if "wl" in opt:ax.plot(  mct, wl, label="wl" )
    if "ARTa" in opt:ax.plot(  mct, ARTa, label="ARTa" )
    if "mct" in opt:ax.plot(  mct, mct, label="mct" )


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

