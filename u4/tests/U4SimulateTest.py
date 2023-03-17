#!/usr/bin/env python
"""
U4SimulateTest.py
====================

"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.fold import Fold, RFold
from opticks.ana.p import * 

MODE = int(os.environ.get("MODE","2"))
if MODE > 0:
    from opticks.ana.pvplt import * 
else:
    pass
pass

axes = 0, 2  # X,Z
H,V = axes 


class U4SimulateTest(RFold):
    """
    """
    def __init__(self, f):

        symbol = f.symbol
        q_ = f.seq[:,0]
        q  = ht.seqhis(q_)  # ht from opticks.ana.p 
        qq = ht.Convert(q_)  # (n,32) int8 : for easy access to nibbles 
        n = np.sum( seqnib_(q_), axis=1 )   
        fk = f.aux[:,:,2,2].view(np.uint32)    ## fakemask : for investigating fakes when FAKES_SKIP is disabled


        qu, qi, qn = np.unique(q, return_index=True, return_counts=True)  
        quo = np.argsort(qn)[::-1]  

        qlim = slice(0,10)
        qtab_ = "np.c_[qn,qi,qu][quo][qlim]" 
        qtab = eval(qtab_)
        qtab_ = qtab_.replace("q","%s.q" % symbol)

        CHECK = getattr( f.photon_meta, 'CHECK', [] )
        CHECK = CHECK[0] if len(CHECK) == 1 else "NO-CHECK"

        LAYOUT = getattr( f.photon_meta, 'LAYOUT', [] )
        LAYOUT = LAYOUT[0] if len(LAYOUT) == 1 else "NO-LAYOUT"

        VERSION = getattr( f.photon_meta, 'VERSION', [] )
        VERSION = int(VERSION[0]) if len(VERSION) == 1 else -1
        SCRIPT = os.environ.get("SCRIPT", "no-SCRIPT") 

        TITLE = "N=%d %s # %s/%s " % (VERSION, SCRIPT, LAYOUT, CHECK )
        ID = "N%d__%s__%s" % (VERSION, LAYOUT, CHECK)



        self.f = f 
        self.q_ = q_
        self.q = q
        self.qq = qq
        self.n = n 
        self.fk = fk

        self.qu = qu
        self.qi = qi
        self.qn = qn
        self.quo = quo
        self.qlim = qlim
        self.qtab_ = qtab_
        self.qtab = qtab 
        self.CHECK = CHECK
        self.LAYOUT = LAYOUT
        self.TITLE = TITLE
        self.ID = ID


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

    FOLD = os.environ.get("FOLD", None)
    log.info(" -- U4SimulateTest.Load FOLD" )
    a = U4SimulateTest.Load(FOLD, symbol="a")   # optional photon histories 
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
        fig, ax = mpplt_plotter(label)

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

