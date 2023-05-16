#!/usr/bin/env python
"""
U4SimulateTest_tt.py
========================

::

    u4t
    ./U4SimulateTest.sh tt
    ./U4SimulateTest.sh ntt

    PLOT=STAMP ~/opticks/u4/tests/tt.sh 
    PLOT=PHO_HIS ~/opticks/u4/tests/tt.sh 
    PLOT=PHO_AVG ~/opticks/u4/tests/tt.sh 


"""
import os, numpy as np, textwrap
from opticks.ana.fold import Fold
from opticks.ana.p import * 
from opticks.sysrap.sevt import SEvt


SCRIPT = "./U4SimulateTest.sh tt"
os.environ["SCRIPT"] = SCRIPT 
ENVOUT = os.environ.get("ENVOUT", None)
LABEL = os.environ.get("LABEL", "U4SimulateTest_tt.py" )


N = int(os.environ.get("N", "-1"))
MODE =  int(os.environ.get("MODE", "2"))
assert MODE in [0,2,-2,3,-3]
FLIP = int(os.environ.get("FLIP", "0")) == 1 
TIGHT = int(os.environ.get("TIGHT", "0")) == 1 

PLOT = os.environ.get("PLOT", None)


if MODE != 0:
    from opticks.ana.pvplt import * 
    WM = mp.pyplot.get_current_fig_manager()  
else:
    WM = None
pass

if __name__ == '__main__':

    print("AFOLD:%s" % os.environ.get("AFOLD", "-"))
    print("BFOLD:%s" % os.environ.get("BFOLD", "-"))
    print("N:%d " % N )

    if N == -1:
        a = SEvt.Load("$AFOLD",symbol="a")
        b = SEvt.Load("$BFOLD",symbol="b")
        syms = ['a','b']
        evts = [a,b]
    elif N == 0:
        a = SEvt.Load("$AFOLD",symbol="a")
        b = None
        syms = ['a']
        evts = [a,]
    elif N == 1:
        a = None
        b = SEvt.Load("$BFOLD",symbol="b")
        syms = ['b']
        evts = [b,]
    else:
        assert(0)
    pass

    if not a is None:print(repr(a))
    if not b is None:print(repr(b))


    EXPRS = r"""
    %(sym)s
    %(sym)s.symbol 
    w
    %(sym)s.q[w]
    %(sym)s.n[w]
    np.diff(%(sym)s.tt[w])
    np.c_[%(sym)s.t[w].view("datetime64[us]")] 
    """ 

    for sym in syms:

        n = eval("%(sym)s.n" % locals() ) 
        ww = np.where( n > 24)[0] 
   
        for w in ww:
            for e_ in textwrap.dedent(EXPRS).split("\n"):
                e = e_ % locals()
                print(e) 
                if len(e) == 0 or e[0] == "#": continue
                print(eval(e))
            pass
        pass
    pass


    label = "tt"


    if PLOT == "PHO_HIS":
        fig, axs = mpplt_plotter(nrows=1, ncols=1, label=PLOT, equal=False)
        ax = axs[0]
        e_ = "%(sym)s.s1 - %(sym)s.s0  # PHO_HIS "
        bins = np.logspace( np.log10(0.1),np.log10(500.0), 25 ) 
        h = {} 
        for sym in syms: 
            e = e_ % locals() 
            h[sym] = np.histogram( eval(e) , bins=bins )   
            ax.plot( h[sym][1][:-1], h[sym][0], label=e  )
        pass
        ax.legend()
        fig.show()
    pass


    HEAD = "np.c_["
    TAIL = "]"
    FIELDS = list(filter(None,textwrap.dedent(r"""
    %(sym)s.s1[w] -  %(sym)s.s0[w]                          # beginPhoton->endPhoton
    %(sym)s.t[w,0] - %(sym)s.s0[w]                          # beginPhoton->firstPoint 
    %(sym)s.t[w,%(n_1)s] - %(sym)s.t[w,0]                   # firstPoint->lastPoint 
    %(sym)s.s1[w] - %(sym)s.t[w,%(n_1)s]                    # lastPoint->endPhoton
    """).split("\n")))

    LABELS = list(map(lambda _:_[_.find("#")+1:], FIELDS))
    FIELDS = list(map(lambda _:_[:_.find("#")].strip(), FIELDS))  
    
    print("compare first and last point stamp range with beginPhoton endPhoton range")

    nn = np.arange(2,33)
    for n in nn:  
        n_1 = int(n - 1)
        for sym in syms:
            w_ = "np.where( %(sym)s.n == %(n)s )[0]" % locals()
            w = eval(w_)
            expr__ = "".join([HEAD, ",".join(FIELDS), TAIL ])
            expr_ = expr__ % locals()
            expr = eval(expr_)
            label = " ".join(LABELS)
            if len(expr) == 0: continue 

            print(expr_)
            print(w_)
            print(label)
            print(expr)
        pass
    pass
          

    print(" average photonBegin->photonEnd for different step counts ") 
    ssa = np.zeros((33,len(syms),3), dtype=np.float64 )
    for n in range(33):
        for i,sym in enumerate(syms):
            w_ = "np.where( %(sym)s.n == %(n)s )[0]" % locals()
            w = eval(w_)
            nw = len(w)
            expr_ = "np.average( %(sym)s.ss[%(sym)s.n == %(n)s ])" % locals() 
            ssa[n,i,0] = n
            ssa[n,i,1] = nw
            ssa[n,i,2] = eval(expr_) if nw > 0 else 0.
        pass
    pass

    expr_ = "np.c_[ssa.reshape(-1,6)]"
    print(expr_)
    print(eval(expr_))



    EXPRS_ = r"""
    np.diff(%(sym)s.rr)[0]/1e6                 # Run
    np.diff(%(sym)s.ee)[0]/1e6                 # Evt
    np.sum(%(sym)s.ss)/1e6                     # Pho 
    np.sum(%(sym)s.ss)/np.diff(%(sym)s.ee)[0]  # Pho/Evt
    """
    EXPRS = list(filter(None, textwrap.dedent(EXPRS_).split("\n")))

    rlines = [] 
    for i,sym in enumerate(syms):
        for expr_ in EXPRS:
            label = expr_[expr_.find("#")+1:] if expr_.find("#") > -1 else ""
            expr_ = expr_.split("#")[0].strip()
            expr = expr_ % locals()
            print(expr) 
            val = eval(expr)
            rlines.append("%30s : %8.3f : %s " % ( expr, val, label  ))
        pass
        rlines.append("") 
    pass

    ranno = "\n".join(rlines)

    if PLOT == "PHO_AVG":

        os.environ["RHSANNO"] = ranno
        os.environ["RHSANNO_POS"]="0.45,-0.02"

        label = "PHO_AVG : average beginPhoton->endPhoton CPU time [us] vs (step point count - 2) " 
        print(label)

        fig, axs = mpplt_plotter(nrows=1, ncols=1, label=label, equal=False)
        ax = axs[0]

        #sl = slice(2,16) 
        sl = slice(None) 
        style = {'a':"ro-", 'b':"bo-" }
        fitstyle = {'a':"r:", 'b':"b:" }

        mc = np.zeros( (len(syms), 2), dtype=np.float64 )

        for i,sym in enumerate(syms):
            x_ = ssa[sl,i,0] - 2
            n_ = ssa[sl,i,1]
            y_ = ssa[sl,i,2]
            w = np.where( n_ > 0)[0]
             
            x = x_[w]
            y = y_[w]

            ax.plot( x, y , style[sym], label="%(sym)s : avg(ss) vs n-2  " % locals() )


            wf = np.where( n_ > 10)[0]  ## only fit points with some stats
            x = x_[wf]
            y = y_[wf]

            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            mc[i,0] = m 
            mc[i,1] = c 
            ax.plot(x, m*x + c, fitstyle[sym], label='Fit: m %10.3f c %10.3f ' % (m,c) )
        pass
        ax.legend()
        fig.show()

        print(label)
        expr = "np.c_[mc]"
        print(expr)
        print(eval(expr))

        print("ranno")
        print(ranno)
    pass


    if PLOT == "PHO_N":
        a_n, a_nc = np.unique(a.n, return_counts=True) 
        b_n, b_nc = np.unique(b.n, return_counts=True) 

         
        msg = "consistent when fakes are skipped"
        msg = "much more in A when fakes not skipped"
        label = "PHO_N : A(N=0),B(N=1) photon step counts : %s " % msg  
        fig, axs = mpplt_plotter(nrows=1, ncols=1, label=label, equal=False)
        ax = axs[0]
        ax.set_yscale("log")
        #ax.plot( a_n, a_nc, "ro", label="A"  )
        #ax.plot( b_n, b_nc, "bo", label="B" )

        ax.errorbar( a_n, a_nc,yerr=np.sqrt(a_nc.astype(np.float64)),fmt="ro-", label="A"  )
        ax.errorbar( b_n, b_nc,yerr=np.sqrt(b_nc.astype(np.float64)),fmt="bo-", label="B" )

        ax.legend()
        fig.show()
    pass 




    if PLOT == "PHO_SCAT":

        fig, axs = mpplt_plotter(nrows=1, ncols=1, label=label, equal=False)
        ax = axs[0]

        for i,sym in enumerate(syms):
            x_ = "%(sym)s.n.astype(np.float64)" % locals() 
            y_ = "%(sym)s.ss" % locals() 
            x = eval(x_)
            y = eval(y_)
            if i == 1: x += 0.4

            ax.scatter( x[x<16], y[x<16], s=2 )
        pass
        ax.legend()
        fig.show()
    pass

    if PLOT == "STAMP":

        label = "STAMP : A(left), B(right) : timestamps BeginPhoton,EndPhoton,PointPhoton " 

        fig, axs = mpplt_plotter(nrows=1, ncols=1, label=label, equal=False)
        ax = axs[0]

        #sl = slice(-100,None,1)  # last 100 photons, which are actually processed first 
        #sl = slice(1040,1550,1) 
        sl = slice(None)

        ## a.s0[np.where(a.n > 20)]-a.ee[0]   find time range with big bouncer
        t0 = 162407 - 100 
        t1 = t0 + 3000


        # HMM: need to select based on time stamps, not photon indices
        # a.s0[np.logical_and( a.s0 < a.s0[0], a.s0 > a.s0[0]-100 )]   

        s0_ = {}
        s1_ = {}
        tt_ = {}
        s0 = {}
        s1 = {}
        tt = {} 

        sz = 0.01 
        xx = {'a':[-0.5, -sz], 'b':[sz,0.5] }
        zz = {'a':[-0.2, -sz], 'b':[sz,0.2] }

        for i, sym in enumerate(syms): 
            r0 = eval("%(sym)s.rr[0]" % locals())
            #if i == 1:continue
            print( "sym:%s r0:%10.3f " % (sym, r0))

            s0_[sym] = "%(sym)s.s0[sl] - %(sym)s.ee[0] # beginPhoton " % locals()
            s1_[sym] = "%(sym)s.s1[sl] - %(sym)s.ee[0] # endPhoton " % locals()
            tt_[sym] = "%(sym)s.t[sl][%(sym)s.t[sl]>0] - %(sym)s.ee[0]  # pointPhoton" % locals()

            s0[sym] = eval(s0_[sym])
            s1[sym] = eval(s1_[sym])
            tt[sym] = eval(tt_[sym])

            labs = [s0_[sym], s1_[sym], tt_[sym]]
            cols = "r b g".split()
            qwns = "s0 s1 tt".split()
            assert len(cols) == len(qwns)   
            for j in range(len(qwns)):
                q = qwns[j]
                t = eval("%(q)s[\"%(sym)s\"]" % locals()) 
                xmin,xmax = zz[sym] if q == "tt" else xx[sym]
                ax.hlines( t[np.logical_and(t > t0, t < t1 )], xmin, xmax, cols[j], label=labs[j] )
            pass
        pass

        ax.legend()
        fig.show()
    pass
    if not ENVOUT is None and not PLOT is None:
        envout = "\n".join([
                       "export ENVOUT_PATH=%s" % ENVOUT,
                       "export ENVOUT_CAP_STEM=%s" % PLOT,
                       ""
                       ]) 
        open(ENVOUT, "w").write(envout)
        print(envout)
    pass

 
