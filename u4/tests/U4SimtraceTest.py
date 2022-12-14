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

FOLD = os.environ.get("FOLD", None)

AFOLD = os.environ.get("AFOLD", None)
BFOLD = os.environ.get("BFOLD", None)

APID = int(os.environ.get("APID", -1 ))
BPID = int(os.environ.get("BPID", -1 ))


TOPLINE = os.environ.get("TOPLINE","U4SimtraceTest.py")
BOTLINE = os.environ.get("BOTLINE","%s" % FOLD )
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
        self.fold = fold 
        self.trs_names = trs_names
        self.sfs = sfs

    def mp_geoplot(self, label="U4SimtraceTest"):

        trs = self.fold.trs
        trs_names = self.trs_names
        sfs = self.sfs

        fig, ax = mpplt_plotter(label=label)

        num = len(trs_names)
        for j in range(num):
            i = num - 1 - j  if REVERSE else j
            soname = trs_names[i]
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
        fig.suptitle("\n".join([TOPLINE,BOTLINE,THIRDLINE]))
        fig.show()


    def mp_onephotonplot(self, a): 
        if a is None: return 
        if a.pid == -1: return

        #if "lin" in a.opt:
        if True:
            self.mp_r_lines(a) 
        pass
        if "nrm" in a.opt:
            self.mp_a_normal(a) 
        pass


    def mp_r_lines(self, f): 
        if not hasattr(f,'r'): return
        r = f.r 
        ax = self.ax
        mpplt_add_contiguous_line_segments(ax, r[:,0,:3], axes=(H,V), label=None )
        if "PLAB" in os.environ:
            self.mp_plab(f)
        pass 

    def mp_a_normal(self, f):
        """
        Access customBoundaryStatus char 

        t.aux[261,:32,1,3].copy().view(np.int8)[::4].copy().view("|S32")

        """
        if not hasattr(f,'r'): return
        if not hasattr(f,'a'): return
        ax = self.ax
        r = f.r 
        a = f.a 
        sc = float(os.environ.get("NRM","100"))
        cbs = a[:len(a),1,3].copy().view(np.int8)[::4]
        assert len(cbs) == len(a)

        for i in range(len(a)):
            nrm = a[i,3,:3]  
            pos = r[i,0,:3]   
            npos = pos - sc*nrm/2
            #mpplt_add_line(ax, pos-sc*nrm, pos+sc*nrm, AXES )   
            ax.arrow( npos[H], npos[V], sc*nrm[H], sc*nrm[V], head_width=10, head_length=10, fc='k', ec='k' )
        pass

    def mp_plab(self, f, tweak=False, backgroundcolor=None ):
        """
        point labels
        """
        ax = self.ax
        r = f.r
        a = f.a 
        cbs = a[:len(a),1,3].copy().view(np.int8)[::4]
        hv = r[:,0,(H,V)]

        if backgroundcolor == None:
            backgroundcolor = os.environ.get("BGC", None)
        pass
        for i in range(len(r)):

            #plab = str(i)
            plab = chr(cbs[i])

            dx,dy = 0,0
            if tweak:
                if i==2: dx,dy=-10,0
                if i==3: dx,dy=0,-10
                if i==4: dx,dy=-10,0
                if i==16: dx,dy=10,-10
            pass
            if backgroundcolor is None:
                ax.text(dx+hv[i,0],dy+hv[i,1], plab, fontsize=15 )
            else:
                ax.text(dx+hv[i,0],dy+hv[i,1], plab, fontsize=15, backgroundcolor=backgroundcolor )
            pass
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
        if pid > -1 and hasattr(f,'record') and pid < len(f.record):
            _r = f.record[pid]
            wl = _r[:,2,3]
            r = _r[wl > 0]
            pass
            if hasattr(f,'aux'): 
                _a = f.aux[pid]
                a = _a[wl > 0]
            pass  
        else:
            _r = None
            r = None
            _a = None
            a = None
        pass
        self._pid = pid
        self._r = _r
        self.r = r
        self._a = _a
        self.a = a

    pid = property(_get_pid, _set_pid)




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    t = U4SimtraceTest.Load(FOLD, symbol="t")
    a = U4SimulateTest.Load(AFOLD, symbol="a")
    b = U4SimulateTest.Load(BFOLD, symbol="b")

    if mp:
        t.mp_geoplot()
        t.mp_onephotonplot(a)
        t.mp_onephotonplot(b)
        t.mp_show()
    pass 
pass



