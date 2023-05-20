#!/usr/bin/env python
"""
sevt_tt.py (formerly u4/tests/U4SimulateTest_tt.py)
======================================================

Python script (not module) usage from two bash scripts
--------------------------------------------------------

::

    PLOT=STAMP STAMP_ANNO=1 STAMP_LEGEND=1 ~/opticks/u4/tests/tt.sh 

    PLOT=STAMP STAMP_TT=90000,5000 STAMP_ANNO=1 ~/j/ntds/ntds.sh tt 

    PLOT=PHO_AVG ~/j/ntds/ntds.sh tt 


PLOT envvar selects what to plot
----------------------------------

PLOT=STAMP
   time stamp illustration
 
   STAMP_TT=200000,5000 
   STAMP_TT=200k,5k 
      control time window of STAMP plot in microseconds [us]

   STAMP_ANNO=1  
      enable photon index and history annotation  

PLOT=PHO_AVG
   average beginPhoton->endPhoton CPU time vs photon point count  

PLOT=PHO_HIS
   histogram of beginPhoton->endPhoton CPU times

   PLOT=PHO_HIS ~/j/ntds/ntds.sh tt

PLOT=PHO_N
   A, B photon step point counts (fakes must be skipped in A to get a match)

   PLOT=PHO_N ~/j/ntds/ntds.sh tt


Dev Notes
------------

HMM: for reusability, its tempting to 
just relocate this into sysrap/tests/tt.py 
as a runnable python script which is used from 
wherever with bash level envvars controlling the events 
to be loaded rather than doing python dev to 
bring in functionality at python level. 

When is this bash in control approach appropriate 
as opposed to moving into sevt.py ?  

The level of generality determines what is appropriate
If the functionality really is general and likely 
to be usable from all over the place then it belongs 
into sevt.py with python control and flexibility of use.

However if only likely to use functionality from a few places
pointing at differnt events that doing 
at bash level seems appropriate and avoids python 
module development. 
  
::

    u4t
    ./U4SimulateTest.sh tt

    PLOT=STAMP   ~/opticks/u4/tests/tt.sh 
    PLOT=PHO_HIS ~/opticks/u4/tests/tt.sh 
    PLOT=PHO_AVG ~/opticks/u4/tests/tt.sh 


"""
import os, numpy as np, textwrap, logging
log = logging.getLogger(__name__)
from opticks.ana.fold import Fold
from opticks.ana.p import * 
from opticks.ana.eget import efloatarray_
from opticks.sysrap.sevt import SEvt


SCRIPT = "./U4SimulateTest.sh tt"
os.environ["SCRIPT"] = SCRIPT 
LABEL = os.environ.get("LABEL", "U4SimulateTest_tt.py" )


N = int(os.environ.get("N", "-1"))
MODE =  int(os.environ.get("MODE", "2"))
assert MODE in [0,2,-2,3,-3]
FLIP = int(os.environ.get("FLIP", "0")) == 1 
TIGHT = int(os.environ.get("TIGHT", "0")) == 1 
STAMP_TT = efloatarray_("STAMP_TT", "162307,3000") # "t0,dt"  microseconds 1M=1s 

NEVT = int(os.environ.get("NEVT", 0))  # when NEVT>0 SEvt.LoadConcat loads and concatenates the SEvt
PLOT = os.environ.get("PLOT", None)

ENVOUT = os.environ.get("ENVOUT", None)
CAP_STEM = PLOT
CAP_BASE = None # set below to a.f.base or b.f.base

if MODE != 0:
    from opticks.ana.pvplt import * 
    WM = mp.pyplot.get_current_fig_manager()  
else:
    WM = None
pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("AFOLD:%s" % os.environ.get("AFOLD", "-"))
    print("BFOLD:%s" % os.environ.get("BFOLD", "-"))
    print("N:%d " % N )

    if N == -1:
        a = SEvt.Load("$AFOLD",symbol="a", NEVT=NEVT)
        b = SEvt.Load("$BFOLD",symbol="b", NEVT=NEVT)
        syms = ['a','b']
        evts = [a,b]
    elif N == 0:
        a = SEvt.Load("$AFOLD",symbol="a", NEVT=NEVT)
        b = None
        syms = ['a']
        evts = [a,]
    elif N == 1:
        a = None
        b = SEvt.Load("$BFOLD",symbol="b", NEVT=NEVT)
        syms = ['b']
        evts = [b,]
    else:
        assert(0)
    pass

    ## CAP_BASE is passed via ENVOUT to the invoking bash script
    ## to control where the figs folder with screen captures should be 
    if not a is None:
        CAP_BASE = a.f.base
    elif not b is None:
        CAP_BASE = b.f.base
    pass


    if not a is None:print(repr(a))
    if not b is None:print(repr(b))


    if PLOT == "EXPRS":
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
    pass


    # needs: syms, a, b  

    if PLOT == "PHO_HIS":
        msg = "histogram of beginPhoton->endPhoton durations"
        label = "%s : %s" % (PLOT, msg)
        fig, axs = mpplt_plotter(nrows=1, ncols=1, label=label, equal=False)
        ax = axs[0]
        e_ = "%(sym)s.s1 - %(sym)s.s0  # PHO_HIS "
        
        #bins = np.logspace( np.log10(0.1),np.log10(500.0), 25 ) 
        bins = np.linspace( 0, 500, 50 )

        h = {} 
        for sym in syms: 
            e = e_ % locals() 
            h[sym] = np.histogram( eval(e) , bins=bins )   
            ax.plot( h[sym][1][:-1], h[sym][0], label=e  )
        pass
        ax.legend()
        fig.show()
    pass


    if PLOT == "TABLE":

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


    if PLOT == "PHO_AVG":
        EXPRS_ = r"""
        np.diff(%(sym)s.rr)[0]/1e6                 # Run
        %(sym)s.ee[-1]/1e6                         # Evt
        np.sum(%(sym)s.ss)/1e6                     # Pho 
        np.sum(%(sym)s.ss)/%(sym)s.ee[-1]          # Pho/Evt
        """
        EXPRS = list(filter(None, textwrap.dedent(EXPRS_).split("\n")))

        rlines = [] 
        for i,sym in enumerate(syms):
            for expr_ in EXPRS:
                label = expr_[expr_.find("#")+1:] if expr_.find("#") > -1 else ""
                expr_ = expr_.split("#")[0].strip()
                expr = expr_ % locals()
                print(expr) 
                try:
                    val = eval(expr)
                except ValueError:
                    log.fatal("FAILED EVAL : %s " % expr )
                    val = -1.0
                pass
                rlines.append("%30s : %8.3f : %s " % ( expr, val, label  ))
            pass
            rlines.append("") 
        pass
        ranno = "\n".join(rlines)
    else:
        ranno = "no-ranno"
    pass
 

    if PLOT == "PHO_AVG":

        os.environ["RHSANNO"] = ranno
        os.environ["RHSANNO_POS"]="0.45,-0.02"
        cut = 10 

        label = "PHO_AVG : average beginPhoton->endPhoton duration [us] vs (step point count - 2)   # stat_cut:%d " % cut
        print(label)

        fig, axs = mpplt_plotter(nrows=1, ncols=1, label=label, equal=False)
        ax = axs[0]

        #sl = slice(2,16) 
        sl = slice(None) 
        style = {'a':"ro-", 'b':"bo-" }
        fitstyle = {'a':"r:", 'b':"b:" }

        mc = np.zeros( (len(syms), 2), dtype=np.float64 )


        for i,sym in enumerate(syms):
            x_ = ssa[sl,i,0] - 2   # step point count - 2 (so starts from 0)
            n_ = ssa[sl,i,1]
            y_ = ssa[sl,i,2]
            w = np.where( n_ > cut)[0]
             
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
        msg = "scatter plot of beginPhoton->endPhoton CPU time vs step point count"
        label = "%s : %s" % (PLOT, msg)

        fig, axs = mpplt_plotter(nrows=1, ncols=1, label=label, equal=False)
        ax = axs[0]

        ax.set_ylim(0,1000)

        for i,sym in enumerate(syms):
            x_ = "%(sym)s.n.astype(np.float64)" % locals() 
            y_ = "%(sym)s.ss" % locals() 
            x = eval(x_)
            y = eval(y_)
            if i == 1: x += 0.2

            ax.scatter( x[x<16], y[x<16], s=2 )
        pass
        ax.legend()
        fig.show()
    pass


    """ 
    STAMP NOTES

    HMM: how to annotate the stamp plot with photon indices ?

    For comparibility between A and B it makes more 
    sense to select based on time stamps rather than photon indices. 

    To find time range with big bouncers::
        
         a.s0[np.where(a.n > 20)]-a.ee[0] 

    """
    if PLOT == "STAMP":


        t0 = STAMP_TT[0]
        t1 = STAMP_TT[0]+STAMP_TT[1] 
        stt = os.environ.get("STAMP_TT","-") 

        label = "A:lhs, B:rhs :  STAMP_TT=%s #  (t0,t1):(%d,%d)  " % (stt,t0,t1)

        #os.environ["SUBTITLE"] = subtitle 
        #os.environ["THIRDLINE"] = subtitle 

        fig, axs = mpplt_plotter(nrows=1, ncols=1, label=label, equal=False)
        ax = axs[0]

        #sl = slice(-100,None,1)  # last 100 photons, which are actually processed first 
        #sl = slice(1040,1550,1) 
        sl = slice(None)

        s0_ = {}
        s1_ = {}
        tt_ = {}
        wt_ = {}
        h0_ = {}
        h1_ = {}
        i0_ = {}
        i1_ = {}

        s0 = {}
        s1 = {}
        tt = {} 
        wt = {} 
        h0 = {}
        h1 = {}
        i0 = {}
        i1 = {}

        sz = 0.01 
        xx = {'a':[-0.5, -sz], 'b':[sz,0.5] }
        zz = {'a':[-0.2, -sz], 'b':[sz,0.2] }
        yy = {'a':-0.5, 'b':0.20 }

        qwns = "s0 s1 t h0 h1 i0 i1".split()
        q_expr = "%(sym)s.%(q)s[sl] - %(sym)s.ee[0]" 

        print(" (t0,t1)  (%(t0)d,%(t1)d) " % locals() )

        for i, sym in enumerate(syms): 
            print("sym:%s" % sym)
            for j in range(len(qwns)):
                q = qwns[j]
                expr = q_expr % locals()
                t = eval(expr) 
                w = eval("np.where(np.logical_and(t > t0, t < t1 ))[0]")
                ws = str(w.shape)
                qmn = t.min()
                qmx = t.max()
                fmt = " %(q)2s  %(expr)22s : (%(qmn)8d,%(qmx)8d) : ws:%(ws)s "  
                print( fmt % locals() )
            pass
        pass

        print(" when all the ws are 0: adjust the time range to find some stamps")


        for i, sym in enumerate(syms): 
            r0 = eval("%(sym)s.rr[0]" % locals())
            #if i == 1:continue
            print( "sym:%s r0:%10.3f " % (sym, r0))

            ## where selecting on times greater than zero messes up the indices, 
            ## so instead just rely on exclusion of crazy times for unfilled cases

            cols = "r b g c m c m".split()
            assert len(cols) == len(qwns) 
  
            for j in range(len(qwns)):
                q = qwns[j]
                expr = q_expr % locals()
                t = eval(expr) 
                w = eval("np.where(np.logical_and(t > t0, t < t1 ))[0]")
                # photon indices of times within time window 
                kk = list(range(len(w)))  
                xmin,xmax = zz[sym] if q == "t" else xx[sym]

                ax.hlines( t[np.logical_and(t > t0, t < t1 )], xmin, xmax, cols[j], label=expr )

                if "STAMP_ANNO" in os.environ:
                    for k in kk:
                        idx = w[k]
                        anno = "(%s) %s : %d : " % (q, sym.upper(), idx) 
                        if q == "s0":
                            his = eval("%(sym)s.q[%(idx)s][0].decode(\"utf-8\").strip()" % locals())  
                            anno +=  his 
                        elif q == "h0":
                            hc = eval("%(sym)s.hc[%(idx)s]" % locals())
                            anno += " hc:%d " % (hc)
                        elif q == "i0":
                            ic = eval("%(sym)s.ic[%(idx)s]" % locals())
                            hi0 = eval("%(sym)s.hi0[%(idx)s]" % locals())
                            anno += "ic:%d hi0:%d "% (ic, hi0)
                        else:
                            anno = None
                        pass
                        if not anno is None:
                            ax.text( yy[sym], t[idx], anno )
                        pass
                    pass
                pass
            pass
        pass

        if "STAMP_LEGEND" in os.environ:
            ax.legend()
        pass
        fig.show()
    pass
    if not ENVOUT is None and not CAP_STEM is None and not CAP_BASE is None:
        envout = "\n".join([
                       "export ENVOUT_PATH=%s" % ENVOUT,
                       "export ENVOUT_CAP_STEM=%s" % CAP_STEM,
                       "export ENVOUT_CAP_BASE=%s" % CAP_BASE,
                       ""
                       ]) 
        open(ENVOUT, "w").write(envout)
        print(envout)
    pass

 
