#!/usr/bin/env python
"""
U4SimtraceTest.py
====================

envvars
---------

FOLD, SFOLD
    mandatory first geometry 
TFOLD
    optional second geometry  
AFOLD
    optional first simulation photon histories
BFOLD
    optional second simulation photon histories

classes
---------

RFold
   Fold loader

U4SimtraceTest(RFold)
   intersection geometry  

U4SimulateTest(RFold)
   photon histories 

"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from collections import OrderedDict as odict 
from opticks.ana.fold import Fold
from opticks.ana.pvplt import *
from opticks.ana.p import * 
from opticks.ana.eget import eintarray_  

COLORS = "red green blue cyan magenta yellow pink orange purple lightgreen".split()
GCOL = "grey"

SZ = float(os.environ.get("SZ",3))
REVERSE = int(os.environ.get("REVERSE","0")) == 1
size = np.array([1280, 720])
X,Y,Z = 0,1,2

FOLD = os.environ.get("FOLD", None)
SFOLD = os.environ.get("SFOLD", None)
TFOLD = os.environ.get("TFOLD", None)
AFOLD = os.environ.get("AFOLD", None)
BFOLD = os.environ.get("BFOLD", None)

N = int(os.environ.get("VERSION","-1")) 
GEOM = os.environ.get("GEOM", "DummyGEOM")
GEOMList = os.environ.get("%s_GEOMList" % GEOM, "DummyGEOMList") 

APID = int(os.environ.get("APID", -1 ))
BPID = int(os.environ.get("BPID", -1 ))
AOPT = os.environ.get("AOPT", "")
BOPT = os.environ.get("BOPT", "")

topline = "N=%d " % N
if APID > -1: topline += "APID=%d " % APID
if len(AOPT) > 0: topline += "AOPT=%s " % AOPT

if BPID > -1: topline += "BPID=%d " % BPID
if len(BOPT) > 0: topline += "BOPT=%s " % BOPT

topline += " ./U4SimtraceTest.py ana   # %s/%s " % (GEOM, GEOMList)

TOPLINE = os.environ.get("TOPLINE",topline )
BOTLINE = os.environ.get("BOTLINE","%s" % SFOLD )
THIRDLINE = os.environ.get("THIRDLINE", "APID:%d BPID:%d " % (APID,BPID) )

AXES = np.array(list(map(int,os.environ.get("AXES","0,2").split(","))), dtype=np.int8 )
H,V = AXES

log = logging.getLogger(__name__)
np.set_printoptions(suppress=True, edgeitems=5, linewidth=200,precision=3)


class RFold(object):
    """
    Provides a common Load method for U4SimtraceTest and U4SimulateTest objects
    """
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

        kvol = "%sVOL" % symbol.upper()
        vol = eintarray_(kvol)

        self.fold = fold 
        self.trs_names = trs_names
        self.sfs = sfs
        self.opp = []
        self.kvol = kvol 
        self.vol = vol 
        self.symbol = symbol

    def geoplot(self, pl):
        """
        :param pl: wither tuple of (fig,ax) for MODE 2 or pyvista pl for MODE 3
        """
        fig, ax = None, None
        if MODE == 2:
            fig, ax = pl
        elif MODE == 2:
            pass
        pass 

        trs = self.fold.trs
        trs_names = self.trs_names
        sfs = self.sfs
        vol = self.vol
        kvol = self.kvol

        vsel = len(vol) > 0   # selecting volumes by index from eg SVOL envvar
        num = len(trs_names)

        if vsel:
            print("mp_geoplot : volume selection %s vol %s num %d " % (kvol, str(vol), num )) 
        pass

        for j in range(num):
            i = num - 1 - j  if REVERSE else j

            soname = trs_names[i]

            if vsel:
                if i in vol: 
                    print(" %s PROCEED i %d soname %s " % (kvol, i, soname)) 
                else:
                    print(" %s SKIP    i %d soname %s " % (kvol, i, soname)) 
                    continue
                pass
            pass

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

            if MODE == 2:
                ax.scatter( gpos[:,H], gpos[:,V], s=SZ, color=color, label=label )
            elif MODE == 3:
                pl.add_points( gpos[:,:3], color=color, label=label)  
            pass
        pass

        if MODE == 2:
            xlim, ylim = mpplt_focus_aspect()
            if not xlim is None:
                ax.set_xlim(xlim) 
                ax.set_ylim(ylim) 
            else:
                log.info("mpplt_focus_aspect not enabled, use eg FOCUS=0,0,100 to enable ")
            pass 
        elif MODE == 3:
            pass
        pass

    def show(self, pl ):
        if MODE == 2:
            self.mp_show(pl)
        elif MODE == 3:
            self.pv_show(pl)
        pass
  
    def pv_show(self, pl):
        cp = pl.show()
        print(cp)

    def mp_show(self, pl): 

        fig, ax = pl
        locs = ["upper left","lower left", "upper right"]
        LOC = os.environ.get("LOC",locs[0])
        print("LOC : %s " % LOC)
        if LOC == "skip" or LOC == "":
            print("skip legend as LOC:[%s] " % LOC)
        else:
            ax.legend(loc=LOC,  markerscale=4)
        pass

        thirdline = ""
        subtitle = ""

        num_opp = len(self.opp)

        if num_opp == 2:
            a,b = self.opp
            thirdline = a.label 
            subtitle  = b.label
        else:
            for one in self.opp:
                print("one.label:%s " % one.label)
                thirdline += one.label 
            pass 
        pass

        subtitle = os.environ.get("SUBTITLE", "")
        print("num_opp:%d subtitle:%s " % (num_opp, subtitle) )
        
        # adjust the position of the title, to legibly display 4 lines      
        TOF = float(os.environ.get("TOF","0.99")) 

        #suptitle = "\n".join([TOPLINE,BOTLINE,thirdline]) 
        suptitle = TOPLINE 

        fig.suptitle(suptitle, family="monospace", va="top", y=TOF, fontweight='bold' )

        ax.text(-0.05,  1.02, thirdline, va='bottom', ha='left', family="monospace", fontsize=12, transform=ax.transAxes)
        ax.text( 1.05, -0.12, subtitle, va='bottom', ha='right', family="monospace", fontsize=12, transform=ax.transAxes)

        fig.show()

    def onephotonplot(self, pl, f): 
        """
        :param pl:  plotting machinery 
        :param f: expecting U4SimulateTest object with photon history, 
                  NB pid must be set to +ve integer selecting the photon to plot anything  
        """
        if f is None: return 
        if f.pid < 0: return

        self.opp.append(f)

        if not hasattr(f,'r'): return
        if f.r is None: return

        r = f.r 
        off = f.off

        rpos = r[:,0,:3] + off

        if MODE == 2:
            fig, ax = pl 
            if True:
                mpplt_add_contiguous_line_segments(ax, rpos, axes=(H,V), label=None )
                self.mp_plab(ax, f)
            pass
            if "nrm" in f.opt:
                self.mp_a_normal(ax, f) 
            pass
        elif MODE == 3:
            pass
            pvplt_add_contiguous_line_segments(pl, rpos )
        pass 

    def mp_plab(self, ax, f, backgroundcolor=None ):
        """
        point labels
        """
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



    def mp_a_normal(self, ax, f):
        """
        Access customBoundaryStatus char 

        t.aux[261,:32,1,3].copy().view(np.int8)[::4].copy().view("|S32")

        """
        if not hasattr(f,'r'): return
        if not hasattr(f,'a'): return
        if f.r is None: return

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
    """
    TODO: separate and relocate for easier reuse
    """
    def __init__(self, f):

        symbol = f.symbol
        q_ = f.seq[:,0]
        q  = ht.seqhis(q_)  # ht from opticks.ana.p 
        qq = ht.Convert(q_)  # (n,32) int8 : for easy access to nibbles 
        n = np.sum( seqnib_(q_), axis=1 )   

        qu, qi, qn = np.unique(q, return_index=True, return_counts=True)  
        quo = np.argsort(qn)[::-1]  

        qtab_ = "np.c_[qn,qi,qu][quo]" 
        qtab = eval(qtab_)
        qtab_ = qtab_.replace("q","%s.q" % symbol)


        self.f = f 
        self.q_ = q_
        self.q = q
        self.qq = qq
        self.n = n 

        self.qu = qu
        self.qi = qi
        self.qn = qn
        self.quo = quo
        self.qtab_ = qtab_
        self.qtab = qtab 


        self._r = None
        self.r = None
        self._a = None
        self.a = None
        self._pid = -1

        pid = int(os.environ.get("%sPID" % symbol.upper(), "-1"))
        opt = os.environ.get("%sOPT" % symbol.upper(), "")
        off_ = os.environ.get("%sOFF" % symbol.upper(), "0,0,0")
        off = np.array(list(map(float,off_.split(","))))

        log.info("U4SimulateTest.__init__  symbol %s pid %d opt %s off %s " % (symbol, pid, opt, str(off)) ) 
        self.symbol = symbol
        self.pid = pid
        self.opt = opt 
        self.off = off

    def __repr__(self):
        return "U4SimulateTest symbol %s pid %s opt %s off %s" % ( self.symbol, self.pid, self.opt, str(self.off) ) 
   
    def _get_pid(self):
        return self._pid
    def _set_pid(self, pid):
        f = self.f
        q = self.q
        symbol = self.symbol
        if pid > -1 and hasattr(f,'record') and pid < len(f.record):
            _r = f.record[pid]
            wl = _r[:,2,3]
            r = _r[wl > 0]    ## select wl>0 to avoid lots of record buffer zeros
            pass
            assert hasattr(f,'aux')
            _a = f.aux[pid]
            a = _a[wl > 0]
            ast = a[:len(a),1,3].copy().view(np.int8)[::4]
            pass  
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
            qpid = None
            label = ""
        pass
        self._pid = pid
        self._r = _r
        self.r = r
        self._a = _a
        self.a = a
        self.ast = ast
        self.qpid = qpid
        self.label = label


    pid = property(_get_pid, _set_pid)




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    fold = SFOLD if not SFOLD is None else FOLD
    s = U4SimtraceTest.Load(fold, symbol="s")    # mandatory first geometry 
    assert not s is None

    t = U4SimtraceTest.Load(TFOLD, symbol="t")   # optional second geometry 
    ## SFOLD, TFOLD and s, t correspond to intersection geometries from U4SimtraceTest


    a = U4SimulateTest.Load(AFOLD, symbol="a")   # optional photon histories 
    b = U4SimulateTest.Load(BFOLD, symbol="b")
    ## AFOLD, BFOLD and a, b are photon histories from U4SimulateTest 

    if not a is None: print('a',a.qtab_,"\n",a.qtab)
    if not b is None: print('b',b.qtab_,"\n",b.qtab)


    pl = plotter(label="U4SimtraceTest.py")  # MODE:2 (fig,ax)  MODE:3 pv plotter

    if not pl is None:
        s.geoplot(pl)
        if not t is None: t.geoplot(pl)

        s.onephotonplot(pl, a)   # must set APID to int index of photon for this to do anything
        s.onephotonplot(pl, b)   # must set BPID to int index of photon for this to do anything

        s.show(pl)
    pass 



pass



