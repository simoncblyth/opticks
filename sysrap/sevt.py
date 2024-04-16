#!/usr/bin/env python
"""
sevt.py (formerly opticks.u4.tests.U4SimulateTest)
====================================================


"""
import os, re, logging, textwrap, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.fold import Fold

print("[from opticks.ana.p import * ")
from opticks.ana.p import * 
print("]from opticks.ana.p import * ")

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


class SEvt(object):
    """
    Higher level wrapper for an Opticks SEvt folder of arrays

    WIP: Concatenation of multiple SEvt, starting from Fold concatenation
    """
    @classmethod
    def Load(cls, *args, **kwa):
        NEVT = int(kwa.get("NEVT",0))
        log.info("SEvt.Load NEVT:%d " % NEVT)
        if NEVT > 0:
            f = cls.LoadConcat(*args,**kwa) 
        else:
            f = Fold.Load(*args, **kwa )
        pass
        return None if f is None else cls(f, **kwa)

    @classmethod
    def LoadConcat(cls, *args, **kwa):
        """
        HMM maybe should load the separate SEvt and then concat those, 
        rather than trying to concat at the lower Fold level 
        """
        NEVT = int(kwa.get("NEVT",0))
        assert NEVT > 0
        f = Fold.LoadConcat(*args, **kwa) 
        assert hasattr(f, 'ff') 
        return f 

    def __init__(self, f, **kwa):
        """
        :param f: Fold instance
        :param kwa: override param, only W2M accepted
        """

        self.f = f 
        self.symbol = f.symbol
        self._r = None
        self.r = None
        self._a = None
        self.a = None
        self._pid = -1
        self.W2M = kwa.get("W2M", None)
        print("sevt.init W2M\n", self.W2M )

        self.init_run(f) 
        self.init_meta(f)
        self.init_fold_meta_timestamp(f)
        self.init_fold_meta(f)
        self.init_U4R_names(f)
        self.init_photon(f)
        self.init_record(f)
        self.init_record_sframe(f)
        self.init_SEventConfig_meta(f)
        self.init_seq(f)
        self.init_aux(f) 
        self.init_ee(f)    ## handling of concatenated SEvt is done currently only for ee
        self.init_aux_t(f)
        self.init_sup(f)
        self.init_junoSD_PMT_v2_SProfile(f)
        self.init_env(f)  # setting pid needs frames so has to be late

    @classmethod
    def CommonRunBase(cls, f):
        """
        :param f: Fold instance
        :return urun_base: str
        """
        run_bases = []
        if type(f.base) is str:
            run_base = os.path.dirname(f.base)
            run_bases.append(run_base)
        elif type(f.base) is list:
            for base in f.base:
                run_base = os.path.dirname(base)
                if not run_base in run_bases:
                    run_bases.append(run_base)
                pass  
        else:
            pass
        pass
        assert len(run_bases) == 1 
        urun_base = run_bases[0]
        return urun_base

    def init_run(self, f):
        """
        HMM the run meta should be same for all of concatenated SEvts
        """
        urun_base = self.CommonRunBase(f)
        urun_path = os.path.join( urun_base, "run.npy" )

        # THIS WAS RECURSING AND LOADING THE n001 and p001 folders
        #print("[ sevt.init_run Fold.Load urun_path %s loading urun_base %s " % (urun_path,urun_base) )
        #fp = Fold.Load(urun_base) if os.path.exists(urun_path) else None
        #print("] sevt.init_run Fold.Load ")
        #run_meta = not fp is None and getattr(fp,'run_meta', None) != None
        #self.fp = fp 

        urun_meta = os.path.join( urun_base, "run_meta.txt" )
        run_meta = NPMeta.Load(urun_meta) if os.path.exists(urun_meta) else None
        has_run_meta = not run_meta is None

        prof2stamp_ = lambda prof:prof.split(",")[0]  

        rr_ = [prof2stamp_(run_meta.SEvt__BeginOfRun), 
               prof2stamp_(run_meta.SEvt__EndOfRun  )] if has_run_meta else [0,0]

        rr = np.array(rr_, dtype=np.uint64 )

        self.rr = rr   
        self.run_meta = run_meta

        qwns = ["FAKES_SKIP",]
        for qwn in qwns:
            mval = list(map(int,getattr(run_meta, qwn, ["-1"] ) if has_run_meta else ["-2"])) 
            setattr(self, qwn, mval[0] )
        pass  


    def init_env(self, f):
        """
        """
        pid = int(os.environ.get("%sPID" % f.symbol.upper(), "-1"))
        opt = os.environ.get("%sOPT" % f.symbol.upper(), "")
        off_ = os.environ.get("%sOFF" % f.symbol.upper(), "0,0,0")
        off = np.array(list(map(float,off_.split(","))))
        log.info("SEvt.__init__  symbol %s pid %d opt %s off %s " % (f.symbol, pid, opt, str(off)) ) 

        self.pid = pid   
        self.opt = opt 
        self.off = off

    def init_meta(self, f):
        """
        """
        metakey = os.environ.get("METAKEY", "junoSD_PMT_v2_Opticks_meta" )
        meta = getattr(f, metakey, None)
        self.metakey = metakey
        self.meta = meta 

    def init_fold_meta_timestamp(self, f):
        """
        """
        if getattr(f, "NPFold_meta", None) == None: return  
        msrc = f.NPFold_meta
        if msrc is None: return 

        def mts_(key):
            val = msrc.get_value(key)
            if val == '': val = "0" 
            return np.uint64(val)

        bor = mts_("T_BeginOfRun")
        boe = mts_("t_BeginOfEvent")
        gs0 = mts_("t_setGenstep_0")
        gs1 = mts_("t_setGenstep_1")
        gs2 = mts_("t_setGenstep_2")
        gs3 = mts_("t_setGenstep_3")
        gs4 = mts_("t_setGenstep_4")
        gs5 = mts_("t_setGenstep_5")
        gs6 = mts_("t_setGenstep_6")
        gs7 = mts_("t_setGenstep_7")
        gs8 = mts_("t_setGenstep_8")
        pre = mts_("t_PreLaunch")
        pos = mts_("t_PostLaunch")
        eoe = mts_("t_EndOfEvent")

        b2e = eoe - boe
        ee = np.array([boe, eoe, b2e], dtype=np.uint64 )
        ett = np.array([bor, boe, gs0, gs1, gs2, gs3, gs4, gs5, gs6, gs7, gs8, pre, pos, eoe], dtype=np.uint64 )
        ettl = "bor boe gs0 gs1 gs2 gs3 gs4 gs5 gs6 gs7 gs8 pre pos eoe".split()
        ettd = np.zeros( len(ett), dtype=np.uint64 )
        ettd[1:] = np.diff(ett)

        ettv = ett.view("datetime64[us]")
        ettc = np.c_[ettl, ett, ettv.astype("|S26"), ettd/1e6, ettd]

        self.ee = ee
        self.ett = ett
        self.ettd = ettd
        self.ettv = ettv
        self.ettl = ettl
        self.ettc = ettc 


    def init_fold_meta(self, f):
        """
        """
        if getattr(f, "NPFold_meta", None) == None: return  
        msrc = f.NPFold_meta
        if msrc is None: return 

        GEOM = msrc.get_value("GEOM")
        CHECK = msrc.get_value("CHECK")
        LAYOUT = msrc.get_value("LAYOUT")
        VERSION = msrc.get_value("VERSION")
        if LAYOUT.find(" ") > -1: LAYOUT = ""

        SCRIPT = os.environ.get("SCRIPT", "") 

        GEOMList = msrc.get_value("${GEOM}_GEOMList", "")
        if GEOMList.endswith("GEOMList"): GEOMList = ""

        TITLE = "N=%s %s # %s/%s " % (VERSION, SCRIPT, LAYOUT, CHECK )
        IDE = list(filter(None,[f.symbol.upper(), GEOM, GEOMList, "N%s"%VERSION, LAYOUT, CHECK])) 
        ID = "_".join(IDE)

        self.CHECK = CHECK
        self.LAYOUT = LAYOUT
        self.VERSION = VERSION
        self.SCRIPT = SCRIPT
        self.GEOM = GEOM
        self.GEOMList = GEOMList
        self.TITLE = TITLE
        self.IDE = IDE
        self.ID = ID

    def init_U4R_names(self, f):
        U4R_names = getattr(f, "U4R_names", None)
        SPECS = np.array(U4R_names.lines) if not U4R_names is None else None 
        self.SPECS = SPECS

    def init_photon(self, f):
        if hasattr(f,'photon') and not f.photon is None:
            iix = f.photon[:,1,3].view(np.int32) 
        else:
            iix = None
        pass

        self.iix = iix        

    def init_record(self, f):
        """
        :param f: fold instance
        """
        if hasattr(f,'record') and not f.record is None:

            if f.record.dtype.name == 'float32':
                iir = f.record[:,:,1,3].view(np.int32) 
                idr = f.record[:,:,1,3].view(np.int32)
            elif f.record.dtype.name == 'float64':
                iir = f.record[:,:,1,3].view(np.int64) 
                idr = f.record[:,:,1,3].view(np.int64)
            else:
                assert False 
            pass
        else:
            iir = None
            idr = None
        pass
        self.iir = iir        
        self.idr = idr        

    def init_record_sframe(self, f):

        with_sframe = not getattr(f, "sframe", None) is None
        with_record = not getattr(f, "record", None) is None

        W2M = self.W2M

        if with_sframe and with_record:
            gpos = np.ones( f.record.shape[:-1] )  ## trim last dimension reducing eg (10000,32,4,4) to (10000, 32, 4)
            gpos[:,:,:3] = f.record[:,:,0,:3]      ## point positions of all photons   
            w2m = W2M if not W2M is None else f.sframe.w2m
            lpos = np.dot( gpos, w2m )  ## transform all points from global to local frame     
        else:
            w2m = W2M if not W2M is None else None
            gpos = None
            lpos = None
        pass
        self.w2m = w2m
        self.gpos = gpos 
        self.lpos = lpos 
         


    def init_SEventConfig_meta(self, f):
        meta = getattr(f, 'SEventConfig_meta', {})

        ipl = getattr(meta, "InputPhoton", []) 
        ip = ipl[0] if len(ipl)==1 else None

        ipfl = getattr(meta, "InputPhotonFrame", []) 
        ipf = ipfl[0] if len(ipfl)==1 else None
 
        ipcl = getattr(meta, "InputPhotonCheck", []) 
        ipc = ipc[0] if len(ipcl)==1 else None

        self.ip = ip
        self.ipf = ipf
        self.ipc = ipc

    def q_startswith(self, prefix="TO BT SD", encoding="utf-8"):
        return np.flatnonzero(np.char.startswith(self.q, prefix.encode(encoding) ))

    def q_startswith_or(self, pfx1="TO BT BT BT SA", pfx2="TO BT BT BT SD", encoding="utf-8"):
        return np.flatnonzero(np.logical_or(
                   np.char.startswith(self.q,pfx1.encode(encoding)), 
                   np.char.startswith(self.q,pfx2.encode(encoding)) 
               ))

    def q_startswith_or_(self, pfxs_="TO BT BT BT SA,TO BT BT BT SD", delim=",", encoding="utf-8"):
        """
        Another way to implement this is with np.any 
        """
        pfxs = pfxs_.split(delim)
        aa = []
        for pfx in pfxs:
            aa.append( np.char.startswith(self.q, pfx.encode(encoding)))
        pass
        return np.flatnonzero(np.logical_or.reduce(aa)) 

    def init_seq(self, f):
        """

        +---------------------+----------------------------------------------------------+
        | a.f.seq.shape       | (1000000, 2, 2)                                          |
        +---------------------+----------------------------------------------------------+
        | a.f.seq[:,0].shape  | (1000000, 2)                                             |      
        +---------------------+----------------------------------------------------------+
        | a.f.seq[0,0]        | array([14748335283153718477, 37762157772], dtype=uint64) |
        +---------------------+----------------------------------------------------------+

        """
        symbol = f.symbol
        qlim = QLIM
        qtab_ = "np.c_[qn,qi,qu][quo][qlim]" 

        if hasattr(f,'seq') and not f.seq is None: 
            q_ = f.seq[:,0]
            q  = ht.seqhis(q_)  # ht from opticks.ana.p 
            qq = ht.Convert(q_)  # (n,32) int8 : for easy access to nibbles 
            n = np.sum( seqnib_(q_), axis=1 )  # occupied nibbles, ie number of history points  

            qu, qi, qn = np.unique(q, return_index=True, return_counts=True) 
            quo = np.argsort(qn)[::-1]  

            qtab = eval(qtab_)
            qtab_ = qtab_.replace("q","%s.q" % symbol)

            nosc = np.ones(len(qq), dtype=np.bool )       # start all true
            nosc[np.where(qq == pcf.SC)[0]] = 0     # knock out photons with scatter in their histories

            nore = np.ones(len(qq), dtype=np.bool )       # start all true
            nore[np.where(qq == pcf.RE)[0]] = 0     # knock out photons with reemission in their histories

            noscab = np.ones(len(qq), dtype=np.bool )     # start all true  
            noscab[np.where(qq == pcf.SC)[0]] = 0   # knock out photons with scatter in their histories
            noscab[np.where(qq == pcf.AB)[0]] = 0   # knock out photons with bulk absorb  in their histories
        else:
            q_ = None
            q = None
            qq = None
            n = None
            qu, qi, qn = None, None, None
            quo = None
            qtab = None 

            nosc = None
            nore = None
            noscab = None
        pass

        self.q_ = q_
        self.q = q
        self.qq = qq
        self.n = n 

        self.qu = qu
        self.qi = qi
        self.qn = qn

        self.quo = quo
        self.qtab = qtab 

        self.qlim = qlim
        self.qtab_ = qtab_

        self.nosc = nosc   # mask of photons without SC in their histories
        self.nore = nore   # mask of photons without RE in their histories
        self.noscab = noscab   # mask of photons without SC or AB in their histories



    def minimal_qtab(self, sli="[:10]", dump=False):
        """
        :return qtab: history table in descending count order 
        """
        e = self
        uq,iq,nq = np.unique(e.q, return_index=True, return_counts=True) 
        oq = np.argsort(nq)[::-1]  
        expr ="np.c_[nq,iq,uq][oq]%(sli)s" % locals()  
        qtab = eval(expr)
        if dump:
            log.info("minimal_qtab : %s " % expr)
            print(qtab)
        pass
        return qtab

    def init_aux(self, f):
        """
        Note aux is currently CPU only 
        """
        if hasattr(f, 'aux') and not f.aux is None: 
            fk = f.aux[:,:,2,2].view(np.uint32)    ## fakemask : for investigating fakes when FAKES_SKIP is disabled
            spec_ = f.aux[:,:,2,3].view(np.int32)   ## step spec
            max_point = f.aux.shape[1]   # instead of hardcoding typical values like 32 or 10, use array shape
            uc4 = f.aux[:,:,2,2].copy().view(np.uint8).reshape(-1,max_point,4) ## see sysrap/spho.h c4/C4Pho.h 
            eph = uc4[:,:,1]          # .y    ProcessHits enum at step point level 
            ep = np.max(eph, axis=1 ) #       ProcessHits enum at photon level   
        else:
            fk = None
            spec_ = None
            uc4 = None
            eph = None
            ep = None
            t = None
        pass
        self.fk = fk
        self.spec_ = spec_
        self.uc4 = uc4
        self.eph = eph   # ProcessHits EPH enum at step point level 
        self.ep  = ep    # ProcessHits EPH enum at photon level  

    def init_ee(self, f):
        """
        microsecond[us] timestamps (UTC epoch) [BeginOfEvent,EndOfEvent,Begin2End] 

        For concatenated SEvt the total of all the EndOfEvent-BeginOfEvent 
        is placed into ee[-1]

        TIMING METADATA HAS BEEN MOVED FROM THE PHOTON TO THE FOLD
        with_ff = not getattr(f, 'ff', None) is None 
        log.info("init_ee with_ff:%d" % (with_ff))
        if with_ff:
            kk = f.ff.keys()
            boe = np.uint64(0)
            eoe = np.uint64(0)
            b2e = np.uint64(0)
            for k in kk:
                fk = f.ff[k]
                k_boe = np.uint64(fk.photon_meta.t_BeginOfEvent[0])
                k_eoe = np.uint64(fk.photon_meta.t_EndOfEvent[0])
                k_b2e = np.uint64(k_eoe-k_boe)
                b2e += k_b2e  
            pass
        else:
            boe, eoe, b2e = 0,0,0 
        pass 
        self.ee = np.array([boe, eoe, b2e], dtype=np.uint64 )
        """

    def init_junoSD_PMT_v2_SProfile(self, f):
        """
        The timestamps come from sysrap/sstamp.h and are datetime64[us] (UTC) compliant 

        pf
            uint64_t microsecond timestamps collected by SProfile.h 
        pfr  
            uint64_t (last - first) timestamp difference for each SProfile (currently ProcessHits call)  


        More than 10% of time spent in ProcessHits::

            In [16]: np.sum(a.pfr)/a.ee[-1]
            Out[16]: 0.10450881266649384

            In [17]: np.sum(b.pfr)/b.ee[-1]
            Out[17]: 0.11134881257006096

        """
        if hasattr(f, 'junoSD_PMT_v2_SProfile') and not f.junoSD_PMT_v2_SProfile is None: 
            pf = f.junoSD_PMT_v2_SProfile
            pfmx = np.max(pf[:,1:], axis=1 )
            pfmi = pf[:,1]
            pfr = pfmx - pfmi
        else:
            pf = None 
            pfmx = None
            pfmi = None
            pfr = None
        pass
        self.pf = pf  ## CAUTION: multiple ProcessHits calls per photon, so not in photon index order 
        self.pfmx = pfmx
        self.pfmi = pfmi
        self.pfr  = pfr 



    def init_aux_t(self, f):
        if hasattr(f, 'aux') and not f.aux is None: 
            t = f.aux[:,:,3,:2].copy().view(np.uint64).squeeze()   # step point timestamps 
        else:
            t = None
        pass
        self.t = t      # array of photon step point time stamps (UTC epoch)

    def init_sup(self, f):
        """
        sup is CPU only 

        photon level beginPhoton endPhoton time stamps from the sup quad4 
        """
        if hasattr(f,'sup') and not f.sup is None:
            s0 = f.sup[:,0,:2].copy().view(np.uint64).squeeze()  # SEvt::beginPhoton (xsup.q0.w.x)
            s1 = f.sup[:,0,2:].copy().view(np.uint64).squeeze()  # SEvt::finalPhoton (xsup.q0.w.y)
            ss = s1 - s0      # endPhoton - beginPhoton 

            f0 = f.sup[:,1,:2].copy().view(np.uint64).squeeze()  # SEvt::finalPhoton (xsup.q1.w.x) t_PenultimatePoint 
            f1 = f.sup[:,1,2:].copy().view(np.uint64).squeeze()  # SEvt::finalPhoton (xsup.q1.w.y) t_LastPoint 
            ff = f1 - f0      # LastPoint - PenultimatePoint 

            h0 = f.sup[:,2,:2].copy().view(np.uint64).squeeze()  # SEvt::addProcessHitsStamp(0) (xsup.q2.w.x)  
            h1 = f.sup[:,2,2:].copy().view(np.uint64).squeeze()  # SEvt::addProcessHitsStamp(0) (xsup.q2.w.y)
            hh = h1 - h0                      # timestamp range of SEvt::AddProcessHitsStamp(0) calls
            hc = f.sup[:,3,0].view(np.int32) 

            i0 = f.sup[:,4,:2].copy().view(np.uint64).squeeze()  # SEvt::addProcessHitsStamp(1) (xsup.q4.w.x)  
            i1 = f.sup[:,4,2:].copy().view(np.uint64).squeeze()  # SEvt::addProcessHitsStamp(1) (xsup.q4.w.y)
            ii = i1 - i0                      # timestamp range of SEvt::AddProcessHitsStamp(1) calls
            ic = f.sup[:,5,0].view(np.int32) 

            hi0 = i0 - h0
            hi1 = i1 - h1
        else:
            s0 = None
            s1 = None
            ss = None

            f0 = None
            f1 = None
            ff = None

            h0 = None
            h1 = None
            hh = None
            hc = None

            i0 = None
            i1 = None
            ii = None
            ic = None

            hi0 = None
            hi1 = None
        pass
        if not getattr(self, 't', None) is None and not s0 is None:
            t0 = s0.min()
            tt = self.t.copy() 
            tt[np.where( tt != 0 )] -= t0   # subtract earliest time "pedestal", but leave the zeros 
        else:
            t0 = None
            tt = None
        pass 

        self.t0 = t0    # scalar : event minimum time stamp (which is minimum s0:beginPhoton timestamp)
        self.tt = tt    # array of photon step point time stamps relative to t0 

        self.s0 = s0    # array of beginPhoton time stamps (UTC epoch)
        self.s1 = s1    # array of endPhoton time stamps (UTC epoch)
        self.ss = ss    # array of endPhoton-beginPhoton timestamp differences 

        self.f0 = f0    # array of PenultimatePoint timestamps (UTC epoch)
        self.f1 = f1    # array of LastPoint timestamps (UTC epoch)
        self.ff = ff    # array of LastPoint-PenultimatePoint timestamp differences

        self.h0 = h0    # array of SEvt::AddProcessHitsStamp(0) range begin (UTC epoch)
        self.h1 = h1    # array of SEvt::AddProcessHitsStamp(0) range end (UTC epoch)
        self.hh = hh    # array of SEvt::AddProcessHitsStamp(0) ranges (microseconds [us]) 
        self.hc = hc    # array of SEvt::AddProcessHitsStamp(0) call counts for each photon

        self.i0 = i0    # array of SEvt::AddProcessHitsStamp(1) range begin (UTC epoch)
        self.i1 = i1    # array of SEvt::AddProcessHitsStamp(1) range end (UTC epoch)
        self.ii = ii    # array of SEvt::AddProcessHitsStamp(1) ranges (microseconds [us]) 
        self.ic = ic    # array of SEvt::AddProcessHitsStamp(1) call counts for each photon

        self.hi0 = hi0  # array of range begin SEvt::AddProcessHitsStamp(1)-SEvt::AddProcessHitsStamp(0)  
        self.hi1 = hi1  # array of range end   SEvt::AddProcessHitsStamp(1)-SEvt::AddProcessHitsStamp(0) 

    def __repr__(self):
        fmt = "SEvt symbol %s pid %s opt %s off %s %s.f.base %s " 
        return fmt % ( self.symbol, self.pid, self.opt, str(self.off), self.symbol, self.f.base ) 
   
    def _get_pid(self):
        return self._pid
    def _set_pid(self, pid):
        """
        """
        f = self.f
        q = self.q
        W2M = self.W2M
        symbol = self.f.symbol
        if pid > -1 and hasattr(f,'record') and pid < len(f.record):
            _r = f.record[pid]
            wl = _r[:,2,3]
            r = _r[wl > 0]    ## select wl>0 to avoid lots of record buffer zeros, example shape (13, 4, 4)

            g = np.ones([len(r),4])
            g[:,:3] = r[:,0,:3]  

            w2m = W2M if not W2M is None else f.sframe.w2m
            l = np.dot( g, w2m )
            ld = np.diff(l,axis=0) 

            pass
            _aux = getattr(f, 'aux', None)
            _a = None if _aux is None else _aux[pid]
            if _a is None:
                a = None
                ast = "no-ast"
            else:
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

            g = None
            l = None
            ld = None
        pass
        self._pid = pid
        self._r = _r
        self.r = r
        self._a = _a
        self.a = a
        self.ast = ast
        self.qpid = qpid
        self.label = label
        self.g = g 
        self.l = l 
        self.ld = ld

    pid = property(_get_pid, _set_pid)


    #def _get_w2m(self):
    #    return self._w2m 
    #
    #def _set_w2m(self, w2m=None):
    #    self._w2m = w2m if not w2m is None else f.sframe.w2m 




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
    """
    Comparison of pairs of SEvt 
    """

    def __getitem__(self, sli):
        self.sli = sli
        return self  

    def __init__(self, a, b): 

        self.sli = slice(None)
        log.info("[")
        if a.q is None or b.q is None: 
            qcf = None
            qcf0 = None
        else:
            qcf = QCF( a.q, b.q, symbol="qcf")
            qcf0 = QCFZero(qcf) if "ZERO" in os.environ else None
        pass
        if a.meta is None or b.meta is None:
            meta = None
        else:
            meta = NPMeta.Compare( a.meta, b.meta  )
        pass

        self.a = a 
        self.b = b 
        self.qcf = qcf
        self.qcf0 = qcf0
        self.meta = meta
        log.info("]")

    def q_startswith(self, prefix="TO BT BT SA"):
        """
        This is a cheating way to find accidentally random aligned photons
        for debugging purposes. 
        It is especially useful with eg rain_point storch.h where all photons
        start with same position and direction. 
        """
        return np.where( np.logical_and( self.a.q == self.b.q, np.char.startswith(self.a.q, prefix.encode("utf-8")) ))   

    def __repr__(self):
        ab = self
        a = self.a
        b = self.b
        qcf = self.qcf
        qcf0 = self.qcf0
        meta = self.meta

        lines = []
        lines.append("SAB")

        if not "BRIEF" in os.environ:
            lines.append(str(a))
            lines.append(repr(a.f))
            lines.append(str(b))
            lines.append(repr(b.f))
            if not qcf is None:
                lines.append("ab.qcf.aqu")
                lines.append(repr(qcf.aqu))
                lines.append("ab.qcf.bqu")
                lines.append(repr(qcf.bqu))
            pass
        pass
        lines.append("a.CHECK : %s " % a.CHECK)
        lines.append("b.CHECK : %s " % b.CHECK)
        if not ab.qcf is None:
            lines.append("ab.qcf[:40]")
            lines.append(repr(ab.qcf[:40]))
        pass
        if not ab.qcf0 is None:
            lines.append("ab.qcf0[:30])")
            lines.append(repr(ab.qcf0[:30]))
        pass
        if not meta is None:
            lines.append(str(meta))
        pass
        return "\n".join(lines)




class KeyIdx(object):
    def __init__(self, kk, symbol="msab.ik"):
        self.kk = kk
        self.symbol = symbol
    def __getattr__(self, k):
        wk = np.where( self.kk == k)[0]
        return wk[0] if len(wk) == 1 else -1 
    def __repr__(self):
        kk = self.kk 
        symbol = self.symbol
        ii = np.arange(len(kk))  
        lines = []
        lines.append("KeyIdx %s" % symbol)
        for i, k in enumerate(kk):
            lines.append("%s.%s = %d " % (symbol, k, i)) 
        pass
        return "\n".join(lines)
 
class MSAB(object):
    """
    Comparison of metadata across NEVT pairs of SEvt
    Typically NEVT is small, eg 10 
    """

    EXPRS = r"""

    %(sym)s.c_itab[%(sym)s.ik.YSAVE,1].sum()/%(sym)s.c_itab[%(sym)s.ik.YSAVE,0].sum()  # N=1/N=0 FMT:%%10.4f

    %(sym)s.c_itab[%(sym)s.ik.YSAVE,0].sum()/%(sym)s.c_itab[%(sym)s.ik.YSAVE,1].sum()  # N=0/N=1 FMT:%%10.4f

    %(sym)s.c_ftab[0,1].sum()/%(sym)s.c_ftab[0,0].sum()  # ratio of duration totals N=1/N=0 FMT:%%10.4f

    np.diff(%(sym)s.c_ttab)/1e6   # seconds between event starts

    np.diff(%(sym)s.c_ttab)[0,1].sum()/np.diff(%(sym)s.c_ttab)[0,0].sum() # ratio N=1/N=0 FMT:%%10.4f

    np.c_[%(sym)s.c_ttab[0].T].view('datetime64[us]') # SEvt start times (UTC)

    %(sym)s.c2tab  # c2sum, c2n, c2per for each event

    %(sym)s.c2tab[0,:].sum()/%(sym)s.c2tab[1,:].sum() # c2per_tot FMT:%%10.4f

    """

    def __init__(self, NEVT, AFOLD, BFOLD, symbol="%(sym)s"):

        efmt = "%0.3d"
        assert efmt in AFOLD and efmt in BFOLD
        self.symbol = symbol
        self.exprs = list(filter(None,textwrap.dedent(self.EXPRS).split("\n")))

        itabs = []
        ftabs = []
        ttabs = []

        first = True
        kk0 = None
        skk0 = None 
        ffield0 = None
        ifield0 = None
        tfield0 = None
        c2tab = np.zeros( (3, NEVT)  )

        mab = {}

        print("NEVT:%d" % NEVT)
        assert( NEVT > 0 )

        for i in range(NEVT):

            afold = AFOLD % i
            bfold = BFOLD % i
            a = SEvt.Load(afold,symbol="a", quiet=True)
            b = SEvt.Load(bfold,symbol="b", quiet=True)
            if a is None:
                log.fatal("FAILED TO LOAD SEVT A FROM AFOLD: %s " % afold )
            pass
            if b is None:
                log.fatal("FAILED TO LOAD SEVT B FROM BFOLD: %s " % bfold )
            pass

            ab = SAB(a,b)
            mab[i] = ab 


            kk = ab.meta.kk
            #print("ab.meta.kk (A,B common keys):%s" % str(kk))

            skk = ab.meta.skk
            tab = ab.meta.tab

            # metadata fields that look like floats, integers, timestamps 
            ffield = np.unique(np.where(np.char.find(tab,'.') > -1 )[0])  
            ifield = np.unique(np.where( np.logical_and( np.char.str_len( tab ) < 15, np.char.find(tab, '.') == -1 ))[0])
            tfield = np.unique(np.where( np.logical_and( np.char.str_len( tab ) > 15, np.char.find(tab, '.') == -1 ))[0])

            if first:
                first = False
                skk0 = skk
                kk0 = kk
                tab0 = tab
                ffield0 = ffield
                ifield0 = ifield
                tfield0 = tfield
            else:
                assert np.all( skk == skk0 )
                assert kk == kk0
                assert tab.shape == tab0.shape
                assert np.all( ffield == ffield0 )
                assert np.all( ifield == ifield0 )
                assert np.all( tfield == tfield0 )
            pass

            itab = tab[ifield]
            ftab = tab[ffield]
            ttab = tab[tfield]

            itabs.append(itab)
            ftabs.append(ftab)
            ttabs.append(ttab)

            if not ab.qcf is None:
                c2tab[0,i] = ab.qcf.c2sum
                c2tab[1,i] = ab.qcf.c2n
                c2tab[2,i] = ab.qcf.c2per
            pass

            # hmm how to present together with c2sum, c2n, c2per ?
            if "c2desc" in os.environ:
                fmt = " %s : %%s " % efmt 
                print( fmt % ( i, ab.qcf.c2desc))
            else:
                if "DUMP" in os.environ:print(repr(ab))
            pass
        pass


        self.c2tab = c2tab
        self.c2tab_zero = np.all( c2tab == 0. )

        self.mab = mab 
        akk = np.array( list(kk0)) 
        ikk = akk[ifield]
        fkk = akk[ffield]
        tkk = akk[tfield]

        ik = KeyIdx(ikk, symbol="%s.ik" % self.symbol )
        fk = KeyIdx(fkk, symbol="%s.fk" % self.symbol )
        tk = KeyIdx(tkk, symbol="%s.tk" % self.symbol )
       
        c_itab = np.zeros( (itab.shape[0], itab.shape[1], NEVT ), dtype=np.uint64 )
        for i in range(len(itabs)): 
            c_itab[:,:,i] = itabs[i]  
        pass
        c_ftab = np.zeros( (ftab.shape[0], ftab.shape[1], NEVT ), dtype=np.float64 )
        for i in range(len(ftabs)): 
            c_ftab[:,:,i] = ftabs[i]  
        pass
        c_ttab = np.zeros( (ttab.shape[0], ttab.shape[1], NEVT ), dtype=np.uint64 )
        for i in range(len(ttabs)): 
            c_ttab[:,:,i] = ttabs[i]  
        pass
       
        self.ikk = ikk
        self.fkk = fkk
        self.tkk = tkk

        self.ik = ik
        self.fk = fk
        self.tk = tk

        self.c_itab = c_itab
        self.c_ftab = c_ftab
        self.c_ttab = c_ttab



    def annotab(self, tabsym, keys, nline=3):
        """
        :param tabsym: attribute symbol of table
        :param keys: array of names for each table item
        :param nline: number of lines for each table item 
        :return str: annoted table string 

        Annotate a numpy array repr with keys 
        TODO: split off into reusable AnnoTab? object 
        """
        expr = "%%(sym)s.%s" % tabsym 
        lines = []
        lines.append("\n%s\n" % ( expr % {'sym':self.symbol} ))
        rawlines = repr(eval(expr % {'sym':"self" })).split("\n")
        for i, line in enumerate(rawlines):
            j = i//nline 
            anno = "%2d:%s" % (j,keys[j]) if i % nline == 0 else ""
            lines.append("%s       %s " % (line, anno ))
        pass
        return "\n".join(lines)

    def __repr__(self):
        lines = []
        lines.append(self.annotab("c_itab", self.ikk, 3)) 
        lines.append(self.annotab("c_ftab", self.fkk, 3)) 
        pass
        fmt_ptn = re.compile("FMT:(.*)\s*")
 
        for expr in self.exprs:
            fmt_match = fmt_ptn.search(expr)
            fmt = fmt_match.groups()[0].replace("%%","%") if not fmt_match is None else None
            if "c2tab" in expr and self.c2tab_zero: continue
            lines.append("\n%s\n" % ( expr % {'sym':self.symbol} ))
            value = eval( expr % {'sym':"self"} )
            urep = repr(value) if fmt is None else fmt % value  
            lines.append(urep)
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

