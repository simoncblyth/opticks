#!/usr/bin/env python
"""
NPFold_profile_test.py
=======================

::

    In [8]: a.profile[:,:,0] - a.profile[:,0,0, np.newaxis]
    Out[8]:
    array([[    0,   116, 33834],
           [    0,   100, 14633],
           [    0,   105, 14500],
           [    0,   100, 14586],
           [    0,   107, 14930],
           [    0,   103, 14525],
           [    0,   106, 14541],
           [    0,   101, 14728],
           [    0,   105, 14497],
           [    0,   107, 14511]])
              BOE0  BOE1  EOE



"""
import os, numpy as np
from np.fold import Fold
from np.npmeta import NPMeta

MODE =  int(os.environ.get("MODE", "2"))
PICK =  os.environ.get("PICK", "CF")
PLOT =  os.environ.get("PLOT", "PWE")
TLIM =  np.array(list(map(int,os.environ.get("TLIM", "0,0").split(","))),dtype=np.int32)
QWN = os.environ.get("QWN", "vm")

US, VM, RS = 0, 1, 2 

# https://matplotlib.org/stable/gallery/color/named_colors.html
palette = ["red","green", "blue", 
           "cyan", "magenta", "yellow", 
           "tab:orange", "tab:pink", "tab:olive",
           "tab:purple", "tab:grey", "tab:cyan"
           ]


COLORS = {
   'A':"red", 
   'A0':"pink",
   'A1':"tab:orange",
   'A2':"purple",
   'B':"blue",
   'B0':"blue",
   'B1':"lightblue",
   'B2':"tab:cyan" 
   }

if MODE != 0:
    from opticks.ana.pvplt import * 
pass


class ProfileWithinEvent(object):
    """
    Timeline is folded to present info for multiple 
    events together using times relative to start of each event. 
    """
    def __init__(self, f, symbol="A"):

        lab = f.labels_names   # of the timestamps
        prof = f.subprofile
        meta = f.subprofile_meta

        slab = list(map(NPMeta.Summarize, lab))
        base = meta.base.replace("/data/blyth/opticks/GEOM/", "")
        smry = meta.smry("GPUMeta,prefix,creator")
        sfmt = meta.smry("stampFmt") 
        titl = "%s:ProfileWithinEvent %s " % (symbol, sfmt) 
        title = " ".join([titl,base,smry]) 

        t = prof[:,:,0] - prof[:,0,0, np.newaxis] 

        self.prof = prof
        self.lab = lab
        self.slab = slab
        self.title = title
        self.t = t 
        self.f = f 
        self.symbol = symbol
        print(repr(self))

    def __repr__(self):
        return "\n".join([self.title, "%s.t" % self.symbol, repr(self.t)])

    def plot(self):
        t = self.t
        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=self.title, equal=False)
            ax = axs[0]

            if TLIM[1] > TLIM[0]:
                ax.set_xlim(*TLIM)
            pass
            for i in range(len(t)):
                ax.vlines( t[i,:], i-0.5, i+0.5  ) 
            pass
            ax.legend()
            fig.show()
        pass  
        return ax

    @classmethod
    def ABPlot(cls, a, b):
        """
        PICK=CF PLOT=PWE ~/np/tests/NPFold_profile_test.sh ana
        """
        A = cls(a, symbol="A")
        B = cls(b, symbol="B") 

 
        avB = np.average(B.t[1:,-1])
        avA = np.average(A.t[1:,-1])

        BOA = B.t[:,-1]/A.t[:,-1]  
        _BOA = ( "%4.1f " * len(BOA) ) % tuple(BOA) 
        avBOA = np.average(BOA[1:]) 
        _avBOA = " avg(BOA[1:]) %4.1f " % avBOA
        sBOA = "BOA : %s %s " % (_BOA, _avBOA)
        QQ = [A,B]

        title = "\n".join(["Profile.ABPlot", A.title, B.title, sBOA])
        print(title)

        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=2, label=title, equal=False)
            for p in range(len(axs)):
                ax = axs[p]
                Q = QQ[p]

                if TLIM[1] > TLIM[0]:
                    ax.set_xlim(*TLIM)
                pass
                t = Q.t
                for i in range(len(t)): 
                    ax.vlines( t[i,:], i-0.5, i+0.5 )
                pass
            pass
            ax.legend()
            fig.show()
        pass  
        return A, B


class Profile(object):
    def __init__(self, f, symbol="A", offset=0 ):
        meta = f.profile_meta
        base = f.profile_meta.base.replace("/data/blyth/opticks/GEOM/", "")
        smry = f.profile_meta.smry("GPUMeta,prefix,creator")
        titl = "%s:Profile " % symbol 
        title = " ".join([titl,base,smry]) 

        pr = f.profile[1:]   ## skip the initialization jump 
        color = COLORS[symbol] 

        j = int(symbol[-1]) if symbol[-1].isnumeric() else None

        if j is None:
            # dont distinguish between profile stamps, treat them all the same
            us = pr[:,:,US].ravel() 
            vm = pr[:,:,VM].ravel()
            rs = pr[:,:,RS].ravel()
        else:
            # pick j-th profile stamp eg BeginOfEvent 
            us = pr[:,j,US] 
            vm = pr[:,j,VM]
            rs = pr[:,j,RS]
        pass

        us = (us - us.min())     # us from first stamp 
        vm = vm/1e6 + offset     # GB 
        rs = rs/1e6 + offset     # GB

        self.f = f    
        self.title = title
        self.pr = pr 
        self.us = us
        self.vm = vm              
        self.rs = rs              
        self.symbol = symbol 
        self.color = color 

    def plot(self):
        p = self
        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=p.title, equal=False)
            ax = axs[0]

            if TLIM[1] > TLIM[0]:
                ax.set_xlim(*TLIM)
            pass
            assert QWN in ["vm", "rs"]
            qwn = getattr(p, QWN)      
            #ax.scatter(p.us, qwn, label=QWN) 
            ax.vlines( p.us, qwn-0.01, qwn+0.01, label=QWN ) 

            pass
            ax.legend()
            fig.show()
        pass  
        return ax

    @classmethod
    def ABPlot(cls, a, b):

        A = cls(a, symbol="A")
        B = cls(b, symbol="B") 

        A0 = cls(a, symbol="A0", offset=0.01)
        A1 = cls(a, symbol="A1", offset=0.01)
        A2 = cls(a, symbol="A2", offset=0.01)

        B0 = cls(b, symbol="B0")
        B1 = cls(b, symbol="B1")
        B2 = cls(b, symbol="B2")
    
        #QQ = [A0, B0]
        #QQ = [A1, B1]
        QQ = [A2, B2]
        #QQ = [A0,A1,A2,B0,B1,B2]
        #QQ = [A,B]

        title = "\n".join(["Profile.ABPlot", A.title, B.title])
        print(title)

        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=title, equal=False)
            ax = axs[0]

            if TLIM[1] > TLIM[0]:
                ax.set_xlim(*TLIM)
            pass
            assert QWN in ["vm", "rs"]
      
            for Q in QQ:
                qwn = getattr(Q, QWN)
                label = "%s:%s" % (Q.symbol, QWN)

                j = int(Q.symbol[-1]) if Q.symbol[-1].isnumeric() else -1 
                qd = 0.005 if j == -1 else 0.002 + 0.001*j 
                ax.vlines( Q.us, qwn-qd, qwn+qd, label=label, color=Q.color ) 
                ax.scatter(Q.us, qwn, label=label, color=Q.color ) 
            pass
            ax.legend()
            fig.show()
        pass  
        return A,B

if __name__ == '__main__':
    ab = Fold.Load(symbol="ab")
    print(repr(ab))
    a = ab.a
    b = ab.b 
    have_both = not a is None and not b is None
    ap = a.profile if not a is None else None 
    bp = b.profile if not b is None else None

    yn_ = lambda _:"NO " if _ is None else "YES"
    print("PICK:%s a:%s b:%s  have_both:%s " % (PICK, yn_(a), yn_(b), have_both ))

    if PICK == "CF" and not have_both:
        print("PICK=CF requires both a and b to exist, use PICK=A or PICK=B if otherwise" )
    elif PICK == "CF":
        if PLOT == "PRO":
            A,B = Profile.ABPlot(a, b)
        elif PLOT == "PWE":
            A,B = ProfileWithinEvent.ABPlot(a,b)
        else:
            A,B = None,None
        pass
    elif PICK in ["AB", "BA", "A", "B"]:
        for symbol in PICK:
            sym = symbol.lower()
            e = getattr(ab, sym, None)
            if e is None:
                print("%s:SKIP as MISSING" % sym )
                continue
            pass
            if PLOT == "PRO":
                p = Profile(e, symbol=symbol)
            elif PLOT == "PWE":
                p = ProfileWithinEvent(e, symbol=symbol)
            else:
                p = None
            pass 
            ax = p.plot() if not p is None else None
        pass
    pass



