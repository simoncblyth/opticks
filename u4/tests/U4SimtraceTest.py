#!/usr/bin/env python
"""
U4SimtraceTest.py
====================




"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from collections import OrderedDict as odict 
from opticks.ana.fold import Fold
from opticks.ana.pvplt import mpplt_add_contiguous_line_segments, mpplt_add_line, mpplt_plotter, mpplt_focus_aspect
from opticks.ana.p import * 

COLORS = "red green blue cyan magenta yellow pink orange purple lightgreen".split()
GCOL = "grey"

#from opticks.ana.framegensteps import FrameGensteps
#from opticks.ana.simtrace_positions import SimtracePositions
#from opticks.ana.simtrace_plot import SimtracePlot, pv, mp

SZ = float(os.environ.get("SZ",3))
REVERSE = int(os.environ.get("REVERSE","0")) == 1
size = np.array([1280, 720])
X,Y,Z = 0,1,2

SFOLD = os.environ.get("SFOLD", None)
TFOLD = os.environ.get("TFOLD", None)

AFOLD = os.environ.get("AFOLD", None)
BFOLD = os.environ.get("BFOLD", None)

APID = int(os.environ.get("APID", -1 ))
BPID = int(os.environ.get("BPID", -1 ))

TOPLINE = os.environ.get("TOPLINE","U4SimtraceTest.py")
BOTLINE = os.environ.get("BOTLINE","%s" % SFOLD )
THIRDLINE = os.environ.get("THIRDLINE", "APID:%d BPID:%d " % (APID,BPID) )

AXES = np.array(list(map(int,os.environ.get("AXES","0,2").split(","))), dtype=np.int8 )
H,V = AXES


log = logging.getLogger(__name__)
np.set_printoptions(suppress=True, edgeitems=5, linewidth=200,precision=3)

try:
    import matplotlib.pyplot as mp  
except ImportError:
    mp = None
pass


class RFold(object):
    @classmethod
    def Load(cls, fold, symbol="x"):
        if not fold is None and os.path.isdir(fold): 
            f = Fold.Load(fold, symbol=symbol )
        else:
            f = None
        pass
        return None if f is None else cls(f)


class U4SimtraceTest(RFold):
    def __init__(self, fold):
        trs_names = np.loadtxt( os.path.join(fold.base, "trs_names.txt"), dtype=np.object )
        sfs = odict()
        for i, name in enumerate(trs_names):
            sfs[name] = Fold.Load(fold.base, name, symbol="%s%0.2d" % (fold.symbol,i) )
        pass

        symbol = fold.symbol
        vol = np.array(list(map(int,list(filter(None, os.environ.get("%sVOL" % symbol.upper(), "").split(",") )) )), dtype=np.int32)  

        self.fold = fold 
        self.trs_names = trs_names
        self.sfs = sfs
        self.onephotonplot = []
        self.vol = vol 
        self.symbol = symbol


    def mp_geoplot(self, label="U4SimtraceTest"):

        trs = self.fold.trs
        trs_names = self.trs_names
        sfs = self.sfs
        vol = self.vol

        fig, ax = mpplt_plotter(label=label)

        num = len(trs_names)
        for j in range(num):
            i = num - 1 - j  if REVERSE else j

            soname = trs_names[i]

            if len(vol) > 0 and i in vol: 
                print(" vol skip i %d vol %s num %d soname %s " % (i,str(vol), num, soname)) 
                continue

            sf = sfs[soname]
            tr = np.float32(trs[i])
            tr[:,3] = [0,0,0,1]   ## fixup 4th column, as may contain identity info

            _lpos = sf.simtrace[:,1].copy()
            _lpos[:,3] = 1

            dori = np.sqrt(np.sum(_lpos[:,:3]*_lpos[:,:3],axis=1))
            lpos = _lpos[dori>0]    # exclude points at local origin, misses 

            gpos = np.dot( lpos, tr )
            pass
            color = COLORS[ i % len(COLORS)]

            label = str(soname)
            if label[0] == "_": label = label[1:]   # seems labels starting "_" have special meaning to mpl, causing problems
            label = label.replace("solid","s")

            ax.scatter( gpos[:,H], gpos[:,V], s=SZ, color=color, label=label )
        pass


        xlim, ylim = mpplt_focus_aspect()
        if not xlim is None:
            ax.set_xlim(xlim) 
            ax.set_ylim(ylim) 
        else:
            log.info("mpplt_focus_aspect not enabled, use eg FOCUS=0,0,100 to enable ")
        pass 

        self.fig = fig
        self.ax = ax
        return fig, ax

    def mp_show(self): 
        fig = self.fig
        ax = self.ax

        locs = ["upper left","lower left", "upper right"]
        LOC = os.environ.get("LOC",locs[0])
        if LOC != "skip":
            ax.legend(loc=LOC,  markerscale=4)
        pass

        thirdline = ""
        for one in self.onephotonplot:
            thirdline += one.label 
        pass 
    
        # adjust the position of the title, to legibly display 4 lines      
        TOF = float(os.environ.get("TOF","0.99")) 
        fig.suptitle("\n".join([TOPLINE,BOTLINE,thirdline]), family="monospace", va="top", y=TOF )
        fig.show()


    def mp_onephotonplot(self, a): 
        if a is None: return 
        if a.pid == -1: return

        self.onephotonplot.append(a)

        #if "lin" in a.opt:
        if True:
            self.mp_r_lines(a) 
        pass
        if "nrm" in a.opt:
            self.mp_a_normal(a) 
        pass


    def mp_r_lines(self, f): 
        if not hasattr(f,'r'): return
        if f.r is None: return
        r = f.r 
        ax = self.ax
        mpplt_add_contiguous_line_segments(ax, r[:,0,:3], axes=(H,V), label=None )
        self.mp_plab(f)

    def mp_plab(self, f, backgroundcolor=None ):
        """
        point labels
        """
        ax = self.ax
        r = f.r
        a = f.a 
        ast = f.ast
        hv = r[:,0,(H,V)]

        if backgroundcolor == None:
            backgroundcolor = os.environ.get("BGC", None)
        pass
        for i in range(len(r)):

            if "idx" in f.opt:
                plab = str(i)
            elif "ast" in f.opt:
                plab = chr(ast[i])
            else:
                plab = None
            pass 
            if plab is None: continue

            dx,dy = 0,0

            if "pdy" in f.opt: dy = 10
            if "ndy" in f.opt: dy = -10

            if backgroundcolor is None:
                ax.text(dx+hv[i,0],dy+hv[i,1], plab, fontsize=15 )
            else:
                ax.text(dx+hv[i,0],dy+hv[i,1], plab, fontsize=15, backgroundcolor=backgroundcolor )
            pass
        pass



    def mp_a_normal(self, f):
        """
        Access customBoundaryStatus char 

        t.aux[261,:32,1,3].copy().view(np.int8)[::4].copy().view("|S32")

        """
        if not hasattr(f,'r'): return
        if not hasattr(f,'a'): return
        if f.r is None: return
       

        ax = self.ax
        r = f.r 
        a = f.a 
        sc = float(os.environ.get("NRM","80"))
        #cbs = a[:len(a),1,3].copy().view(np.int8)[::4]
        #assert len(cbs) == len(a)

        ast = f.ast 
        assert len(ast) == len(a) == len(r)


        for i in range(len(a)):
            nrm = a[i,3,:3]  
            pos = r[i,0,:3]   
            npos = pos - sc*nrm/2
            #mpplt_add_line(ax, pos-sc*nrm, pos+sc*nrm, AXES )   
            ax.arrow( npos[H], npos[V], sc*nrm[H], sc*nrm[V], head_width=10, head_length=10, fc='k', ec='k' )
        pass


class U4SimulateTest(RFold):
    def __init__(self, f):
        self.f = f 

        self._r = None
        self.r = None
        self._a = None
        self.a = None
        self._pid = -1

        symbol = f.symbol
        pid = int(os.environ.get("%sPID" % symbol.upper(), "-1"))
        opt = os.environ.get("%sOPT" % symbol.upper(), "")

        log.info("U4SimulateTest.__init__  symbol %s pid %d opt %s " % (symbol, pid, opt) ) 
        self.symbol = symbol
        self.pid = pid
        self.opt = opt 

    def __repr__(self):
        return "U4SimulateTest symbol %s pid %s opt %s" % ( self.symbol, self.pid, self.opt ) 
   
    def _get_pid(self):
        return self._pid
    def _set_pid(self, pid):
        f = self.f
        symbol = self.symbol
        if pid > -1 and hasattr(f,'record') and pid < len(f.record):
            _r = f.record[pid]
            wl = _r[:,2,3]
            r = _r[wl > 0]
            pass
            assert hasattr(f,'aux')
            _a = f.aux[pid]
            a = _a[wl > 0]
            ast = a[:len(a),1,3].copy().view(np.int8)[::4]
            pass  
            q = ht.seqhis(f.seq[:,0])  # ht from opticks.ana.p 
            qpid = q[pid][0].decode("utf-8").strip()  

            npoi = (len(qpid)+1)//3 
            qkey = " ".join(map(lambda _:"%0.2d"%_, range(npoi)))  

            #label = "%s %sPID:%d\n%s " % (qpid, symbol.upper(), pid, qkey)
            label = "%s\n%s" % (qpid, qkey)
        else:
            _r = None
            r = None
            _a = None
            a = None
            ast = None
            q = None
            qpid = None
            label = ""
        pass
        self._pid = pid
        self._r = _r
        self.r = r
        self._a = _a
        self.a = a
        self.ast = ast
        self.q = q 
        self.qpid = qpid
        self.label = label


    pid = property(_get_pid, _set_pid)




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    s = U4SimtraceTest.Load(SFOLD, symbol="s")
    t = U4SimtraceTest.Load(TFOLD, symbol="t")

    a = U4SimulateTest.Load(AFOLD, symbol="a")
    b = U4SimulateTest.Load(BFOLD, symbol="b")

    if mp:
        if not s is None:
            s.mp_geoplot()
        pass
        if not t is None:
            t.mp_geoplot()
        pass
        s.mp_onephotonplot(a)
        s.mp_onephotonplot(b)
        s.mp_show()
    pass 
pass



