#!/usr/bin/env python
"""
sevt.py (formerly opticks.u4.tests.U4SimulateTest)
====================================================


"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.fold import Fold, RFold
from opticks.ana.p import * 
from opticks.ana.eget import eslice_
from opticks.ana.base import PhotonCodeFlags
from opticks.ana.qcf import QU,QCF,QCFZero
from opticks.ana.npmeta import NPMeta    


pcf = PhotonCodeFlags() 
fln = pcf.fln
fla = pcf.fla


MODE = int(os.environ.get("MODE","2"))
if MODE > 0:
    from opticks.ana.pvplt import * 
else:
    pass
pass

QLIM=eslice_("QLIM", "0:10")


axes = 0, 2  # X,Z
H,V = axes 


class SEvt(RFold):
    """
    Higher level wrapper for an Opticks SEvt folder of arrays


    """
    def __init__(self, f):

        symbol = f.symbol
        qlim = QLIM
        qtab_ = "np.c_[qn,qi,qu][quo][qlim]" 

        if hasattr(f,'seq') and not f.seq is None: 
            q_ = f.seq[:,0]
            q  = ht.seqhis(q_)  # ht from opticks.ana.p 
            qq = ht.Convert(q_)  # (n,32) int8 : for easy access to nibbles 
            n = np.sum( seqnib_(q_), axis=1 )   

            nosc = np.ones(len(qq), np.bool )       # start all true
            nosc[np.where(qq == pcf.SC)[0]] = 0     # knock out photons with scatter in their histories

            noscab = np.ones(len(qq), np.bool )     # start all true  
            noscab[np.where(qq == pcf.SC)[0]] = 0   # knock out photons with scatter in their histories
            noscab[np.where(qq == pcf.AB)[0]] = 0   # knock out photons with bulk absorb  in their histories

            qu, qi, qn = np.unique(q, return_index=True, return_counts=True)  
            quo = np.argsort(qn)[::-1]  

            qtab = eval(qtab_)
            qtab_ = qtab_.replace("q","%s.q" % symbol)
        else:
            q_ = None
            q = None
            qq = None
            n = None
            nosc = None
            noscab = None
            qu, qi, qn = None, None, None
            quo = None
            qtab = None 
        pass

        if hasattr(f, 'aux') and not f.aux is None: 
            fk = f.aux[:,:,2,2].view(np.uint32)    ## fakemask : for investigating fakes when FAKES_SKIP is disabled
            spec_ = f.aux[:,:,2,3].view(np.int32)   ## step spec
            uc4 = f.aux[:,:,2,2].copy().view(np.uint8).reshape(-1,32,4) ## see sysrap/spho.h c4/C4Pho.h 
            eph = uc4[:,:,1]          # .y    ProcessHits enum at step point level 
            ep = np.max(eph, axis=1 ) #       ProcessHits enum at photon level   
        else:
            fk = None
            spec_ = None
            uc4 = None
            eph = None
            ep = None
        pass


        CHECK = getattr( f.photon_meta, 'CHECK', [] )
        CHECK = CHECK[0] if len(CHECK) == 1 else ""

        LAYOUT = getattr( f.photon_meta, 'LAYOUT', [] )
        LAYOUT = LAYOUT[0] if len(LAYOUT) == 1 else ""
        if LAYOUT.find(" ") > -1: LAYOUT = ""

        VERSION = getattr( f.photon_meta, 'VERSION', [] )
        VERSION = int(VERSION[0]) if len(VERSION) == 1 else -1
        SCRIPT = os.environ.get("SCRIPT", "") 

        GEOM = getattr(f.photon_meta, "GEOM", [])
        GEOM = GEOM[0] if len(GEOM) == 1 else ""

        GEOMList = getattr(f.photon_meta, "${GEOM}_GEOMList", [])
        GEOMList = GEOMList[0] if len(GEOMList) == 1 else ""
        if GEOMList.endswith("GEOMList"): GEOMList = ""

        U4R_names = getattr(f, "U4R_names", None)
        SPECS = np.array(U4R_names.lines) if not U4R_names is None else None 

        spec = SPECS[spec_]


        TITLE = "N=%d %s # %s/%s " % (VERSION, SCRIPT, LAYOUT, CHECK )

        IDE = list(filter(None,[symbol.upper(), GEOM, GEOMList, "N%d"%VERSION, LAYOUT, CHECK])) 
        ID = "_".join(IDE)


        w2m = f.sframe.w2m
        gpos = np.ones( f.record.shape[:-1] )  ## trim last dimension reducing eg (10000,32,4,4) to (10000, 32, 4)
        gpos[:,:,:3] = f.record[:,:,0,:3]      ## point positions of all photons   
        lpos = np.dot( gpos, w2m )  ## transform all points from global to local frame     



        self.f = f 
        self.q_ = q_
        self.q = q
        self.qq = qq
        self.n = n 
        self.fk = fk
        self.uc4 = uc4
        self.eph = eph   # ProcessHits EPH enum at step point level 
        self.ep  = ep    # ProcessHits EPH enum at photon level  
        self.nosc = nosc   # mask of photons without SC in their histories
        self.noscab = noscab   # mask of photons without SC or AB in their histories

        self.qu = qu
        self.qi = qi
        self.qn = qn
        self.quo = quo
        self.qlim = qlim
        self.qtab_ = qtab_
        self.qtab = qtab 
        self.CHECK = CHECK
        self.LAYOUT = LAYOUT
        self.VERSION = VERSION
        self.SCRIPT = SCRIPT
        self.GEOM = GEOM
        self.GEOMList = GEOMList
        self.SPECS = SPECS

        self.TITLE = TITLE
        self.IDE = IDE
        self.ID = ID

        self.spec_ = spec_
        self.spec = spec 

        self.w2m = w2m
        self.gpos = gpos 
        self.lpos = lpos 

        self._r = None
        self.r = None
        self._a = None
        self.a = None
        self._pid = -1

        pid = int(os.environ.get("%sPID" % symbol.upper(), "-1"))
        opt = os.environ.get("%sOPT" % symbol.upper(), "")
        off_ = os.environ.get("%sOFF" % symbol.upper(), "0,0,0")
        off = np.array(list(map(float,off_.split(","))))

        log.info("SEvt.__init__  symbol %s pid %d opt %s off %s " % (symbol, pid, opt, str(off)) ) 
        self.symbol = symbol
        self.pid = pid
        self.opt = opt 
        self.off = off

    def __repr__(self):
        return "SEvt symbol %s pid %s opt %s off %s" % ( self.symbol, self.pid, self.opt, str(self.off) ) 
   
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

    def spec_(self, i):
        """
        ## need np.abs as detected fakes that are not skipped are negated
        In [10]: np.c_[a.spec[5,:a.n[5]],a.SPECS[np.abs(a.spec[5,:a.n[5]])]]
        Out[10]: 
        array([['0', 'UNSET'],
               ['1', 'Water/Water:pInnerWater/pLPMT_Hamamatsu_R12860'],
               ['2', 'Water/AcrylicMask:pLPMT_Hamamatsu_R12860/HamamatsuR12860pMask'],
               ['3', 'AcrylicMask/Water:HamamatsuR12860pMask/pLPMT_Hamamatsu_R12860'],
               ['4', 'Water/Pyrex:pLPMT_Hamamatsu_R12860/HamamatsuR12860_PMT_20inch_log_phys'],
               ['-5', 'Pyrex/Pyrex:HamamatsuR12860_PMT_20inch_log_phys/HamamatsuR12860_PMT_20inch_body_phys'],
               ['6', 'Pyrex/Pyrex:HamamatsuR12860_PMT_20inch_body_phys/HamamatsuR12860_PMT_20inch_body_phys']], dtype='<U94')

        In [1]: a.spec_(5)
        Out[1]: 
        array([['0', 'UNSET'],
               ['1', 'Water/Water:pInnerWater/pLPMT_Hamamatsu_R12860'],
               ['2', 'Water/AcrylicMask:pLPMT_Hamamatsu_R12860/HamamatsuR12860pMask'],
               ['3', 'AcrylicMask/Water:HamamatsuR12860pMask/pLPMT_Hamamatsu_R12860'],
               ['4', 'Water/Pyrex:pLPMT_Hamamatsu_R12860/HamamatsuR12860_PMT_20inch_log_phys'],
               ['-5', 'Pyrex/Pyrex:HamamatsuR12860_PMT_20inch_log_phys/HamamatsuR12860_PMT_20inch_body_phys'],
               ['6', 'Pyrex/Pyrex:HamamatsuR12860_PMT_20inch_body_phys/HamamatsuR12860_PMT_20inch_body_phys']], dtype='<U94')

        In [2]: b.spec_(5)
        Out[2]: 
        array([['0', 'UNSET'],
               ['1', 'Water/Water:pInnerWater/pLPMT_Hamamatsu_R12860'],
               ['2', 'Water/AcrylicMask:pLPMT_Hamamatsu_R12860/HamamatsuR12860pMask'],
               ['3', 'AcrylicMask/Water:HamamatsuR12860pMask/pLPMT_Hamamatsu_R12860'],
               ['4', 'Water/Pyrex:pLPMT_Hamamatsu_R12860/HamamatsuR12860_PMT_20inch_log_phys'],
               ['5', 'Pyrex/Vacuum:HamamatsuR12860_PMT_20inch_log_phys/HamamatsuR12860_PMT_20inch_inner_phys']], dtype='<U93')

        """
        t = self
        n = t.n[i]
        spec = t.spec[i,:n]
        return np.c_[spec,t.SPECS[np.abs(spec)]]


class SAB(object):
    def __init__(self, a, b): 
        qcf = QCF( a.q, b.q, symbol="qcf")
        qcf0 = QCFZero(qcf) if "ZERO" in os.environ else None
        self.a = a 
        self.b = b 
        self.qcf = qcf
        self.qcf0 = qcf0
        self.jsdph = NPMeta.Compare( a.f.junoSD_PMT_v2_meta , b.f.junoSD_PMT_v2_meta )   

    def __repr__(self):
        a = self.a
        b = self.b
        qcf = self.qcf
        qcf0 = self.qcf0
        jsdph = self.jsdph 

        lines = []
        lines.append("SAB")

        if not "BRIEF" in os.environ:
            lines.append(str(a))
            lines.append(repr(a.f))
            lines.append(str(b))
            lines.append(repr(b.f))
            lines.append(repr(qcf.aqu))
            lines.append(repr(qcf.bqu))
        pass
        lines.append(repr(qcf))

        if not qcf0 is None:
            lines.append(repr(qcf0))
        pass
        if not jsdph is None:
            lines.append(str(jsdph))
        pass
        return "\n".join(lines)





if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    FOLD = os.environ.get("FOLD", None)
    log.info(" -- SEvt.Load FOLD" )
    a = SEvt.Load(FOLD, symbol="a")   # optional photon histories 
    print(a)

    beg_ = "a.f.record[:,0,0,:3]"
    beg = eval(beg_)

    end_ = "a.f.photon[:,0,:3]"
    end = eval(end_)


    #label0, ppos0 = None, None
    label0, ppos0 = "b:%s" % beg_ , beg

    #label0, ppos0 = None, None
    label1, ppos1 = "r:%s" % end_ , end


    HEADLINE = "%s %s" % ( a.LAYOUT, a.CHECK )
    label = "\n".join( filter(None, [HEADLINE, label0, label1]))
    print(label)

    if MODE == 0:
        print("not plotting as MODE 0  in environ")
    elif MODE == 2:
        fig, ax = mpplt_plotter(label=label)

        ax.set_ylim(-250,250)
        ax.set_xlim(-500,500)

        if not ppos0 is None: ax.scatter( ppos0[:,H], ppos0[:,V], s=1, c="b" )  
        if not ppos1 is None: ax.scatter( ppos1[:,H], ppos1[:,V], s=1, c="r" )  

        fig.show()
    elif MODE == 3:
        pl = pvplt_plotter(label)
        os.environ["EYE"] = "0,100,165"
        os.environ["LOOK"] = "0,0,165"
        pvplt_viewpoint(pl)
        pl.add_points(ppos0)
        pl.show()
    pass
pass

