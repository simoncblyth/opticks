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


def old_check_test(s):
    """
    """
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

def check_domain(f):
    lpmtid_domain = f.lpmtid_domain
    lpmtcat_domain = f.lpmtcat_domain
    assert len(lpmtid_domain) == len(lpmtcat_domain)

    mct_domain = f.mct_domain
    st_domain = f.st_domain
    assert len(mct_domain) == len(st_domain)

'-', '--', '-.', ':', ''

plotmap = { 
   'As':["r","-"],
   'Ap':["r","--"],
   'Aa':["r","-."], 
   'A_':["r",":"],
   'Rs':["g","-"], 
   'Rp':["g","--"], 
   'Ra':["g","-."], 
   'R_':["g",":"],
   'Ts':["b","-"], 
   'Tp':["b","--"], 
   'Ta':["b","-."], 
   'T_':["b",":"],
   } 


color_     = lambda _:plotmap.get(_,["k","."])[0]    
linestyle_ = lambda _:plotmap.get(_,["k","."])[1]


if __name__ == '__main__':
    s = Fold.Load("$SFOLD/spmt", symbol="s")
    print(repr(s))

    f = Fold.Load("$SFOLD/sscan", symbol="f")
    print(repr(f))

    check_nan(f)
    check_domain(f)

    args = f.args.squeeze()
    spec = f.spec.squeeze()
    extra = f.extra.squeeze()
    ARTE = f.ARTE.squeeze()

    stack = f.stack.squeeze()
    ll = f.ll.squeeze() 
    comp = f.comp.squeeze() 
    art = f.art.squeeze() 

    num_lpmtid = len(f.lpmtid_domain)
    num_mct = len(f.mct_domain) 
    art_shape = (num_lpmtid, num_mct, 4, 4)
    assert art.shape == art_shape, art.shape

    PMTIDX = np.fromstring(os.environ.get("PMTIDX","0"),dtype=np.int64, sep=",") 
    lpmtid = f.lpmtid_domain[PMTIDX]
    lpmtcat = f.lpmtcat_domain[PMTIDX]

    OPT = "A_,R_,T_,As,Rs,Ts,Ap,Rp,Tp,Aa,Ra,Ta"
    opt = os.environ.get("OPT", OPT)


    expr = "np.c_[PMTIDX,lpmtid,lpmtcat].T"
    etab = eval(expr)
    mtab = "PMTIDX %s lpmtid %s lpmtcat %s " % ( str(PMTIDX), str(lpmtid), str(lpmtcat) )
    title = "%s : OPT %s \n%s" % (f.base, opt, etab ) 

    print(title)


    fig, ax = plt.subplots(1, figsize=SIZE/100.)
    fig.suptitle(title)

    for i, pmtidx in enumerate(PMTIDX):
        As   = art[pmtidx,:,0,0]
        Ap   = art[pmtidx,:,0,1]
        Aa   = art[pmtidx,:,0,2]
        A_   = art[pmtidx,:,0,3]

        Rs   = art[pmtidx,:,1,0]
        Rp   = art[pmtidx,:,1,1]
        Ra   = art[pmtidx,:,1,2]
        R_   = art[pmtidx,:,1,3]

        Ts   = art[pmtidx,:,2,0]
        Tp   = art[pmtidx,:,2,1]
        Ta   = art[pmtidx,:,2,2]
        T_   = art[pmtidx,:,2,3]

        SF     = art[pmtidx,:,3,0]
        wl     = art[pmtidx,:,3,1] 
        ARTa   = art[pmtidx,:,3,2]
        mct    = art[pmtidx,:,3,3]

        if i == 0: 
            label_ = lambda _:_
        else:
            label_ = lambda _:None
        pass

        if "As" in opt:ax.plot(  mct, As, label=label_("As"), color=color_("As"), linestyle=linestyle_("As") )
        if "Ap" in opt:ax.plot(  mct, Ap, label=label_("Ap"), color=color_("Ap"), linestyle=linestyle_("Ap"))
        if "Aa" in opt:ax.plot(  mct, Aa, label=label_("Aa"), color=color_("Aa"), linestyle=linestyle_("Aa"))
        if "A_" in opt:ax.plot(  mct, A_, label=label_("A_"), color=color_("A_"), linestyle=linestyle_("A_"))

        if "Rs" in opt:ax.plot(  mct, Rs, label=label_("Rs"), color=color_("Rs"), linestyle=linestyle_("Rs"))
        if "Rp" in opt:ax.plot(  mct, Rp, label=label_("Rp"), color=color_("Rp"), linestyle=linestyle_("Rp"))
        if "Ra" in opt:ax.plot(  mct, Ra, label=label_("Ra"), color=color_("Ra"), linestyle=linestyle_("Ra"))
        if "R_" in opt:ax.plot(  mct, R_, label=label_("R_"), color=color_("R_"), linestyle=linestyle_("R_"))

        if "Ts" in opt:ax.plot(  mct, Ts, label=label_("Ts"), color=color_("Ts"), linestyle=linestyle_("Ts"))
        if "Tp" in opt:ax.plot(  mct, Tp, label=label_("Tp"), color=color_("Tp"), linestyle=linestyle_("Tp"))
        if "Ta" in opt:ax.plot(  mct, Ta, label=label_("Ta"), color=color_("Ta"), linestyle=linestyle_("Ta"))
        if "T_" in opt:ax.plot(  mct, T_, label=label_("T_"), color=color_("T_"), linestyle=linestyle_("T_"))

        if "SF" in opt:ax.plot(  mct, SF, label=label_("SF"), color=color_("SF"), linestyle=linestyle_("SF")) 
        if "wl" in opt:ax.plot(  mct, wl, label=label_("wl"), color=color_("wl"), linestyle=linestyle_("wl") )
        if "ARTa" in opt:ax.plot(  mct, ARTa, label=label_("ARTa"), color=color_("ARTa"), linestyle=linestyle_("ARTa") )
        if "mct" in opt:ax.plot(  mct, mct, label=label_("mct"), color=color_("mct"), linestyle=linestyle_("mct") )
    pass
    ax.legend()
    fig.show()


