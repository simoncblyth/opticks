#!usr/bin/env python
"""

"""
import os, sys, logging, numpy as np
from collections import OrderedDict as odict

from opticks.ana.base import opticks_main
from opticks.ana.ctx import Ctx
from opticks.ana.cfh import CFH
from opticks.ana.nbase import chi2, vnorm
from opticks.ana.decompression import decompression_bins
from opticks.ana.histype import HisType
from opticks.ana.mattype import MatType
from opticks.ana.evt import Evt
from opticks.ana.abstat import ABStat
from opticks.ana.make_rst_table import recarray_as_rst

log = logging.getLogger(__name__)


class AB(object):
    """
    Selection examples::

         ab.sel = ".6ccd"             
         ab.sel = "TO BT BT SC .."     # wildcard selection same as above
         ab.sel = None                 # back to default no selection

    Subsequently check tables with::

         ab.his
         ab.flg
         ab.mat

    Histo persisting, provides random access to histos for debugging::

         ab.sel = slice(0,1)
         ab.qwn = "Z"
         ab.irec = 5

         h = ab.h      # qwn and irec used in creation of histogram, which is persisted

    For example, checking a hi chi2 contributor::

        In [12]: h = ab.h
        [2016-11-14 12:21:43,098] p12883 {/Users/blyth/opticks/ana/ab.py:473} INFO - AB.rhist qwn T irec 5 
        [2016-11-14 12:21:43,098] p12883 {/Users/blyth/opticks/ana/cfh.py:344} INFO - CFH.load from /tmp/blyth/opticks/CFH/concentric/1/TO_BT_BT_BT_BT_AB/5/T 

        In [13]: h.lhabc
        Out[13]: 
        array([[     2.9649,      6.9934,      0.    ,      0.    ,      0.    ],
               [     6.9934,     11.0218,      0.    ,      0.    ,      0.    ],
               [    11.0218,     15.0503,      0.    ,      0.    ,      0.    ],
               [    15.0503,     19.0787,      0.    ,      0.    ,      0.    ],
               [    19.0787,     23.1072,  14159.    ,  13454.    ,     17.9997],
               [    23.1072,     27.1356,  14796.    ,  15195.    ,      5.3083],
               [    27.1356,     31.164 ,      0.    ,      0.    ,      0.    ],
               [    31.164 ,     35.1925,      0.    ,      0.    ,      0.    ],
               [    35.1925,     39.2209,      0.    ,      0.    ,      0.    ],
               [    39.2209,     43.2494,      0.    ,      0.    ,      0.    ]], dtype=float32)

    """
    C2CUT = 30

    def __init__(self, ok):
        self.ok = ok
        self.tabs = []
        self.load()
        self.compare()
        self.init_point()

    def load(self):
        """
        It takes aound 6s to load 1M full AB evt pair. So avoid needing to duplicate that.
        """
        log.info("AB.load START ")
        args = self.ok
        try:
            a = Evt(tag="%s" % args.tag, src=args.src, det=args.det, args=args, nom="A")
            b = Evt(tag="-%s" % args.tag, src=args.src, det=args.det, args=args, nom="B")
        except IOError as err:
            log.fatal(err)
            sys.exit(args.mrc)
        pass
        self.a = a
        self.b = b 
        self._dirty = False
        ## property setters
        self.sel = None
        self.irec = 0
        self.qwn = "X"
        log.info("AB.load DONE ")

    def __repr__(self):
        abn = "AB(%s,%s,%s)  %s %s " % (self.ok.tag, self.ok.src, self.ok.det, self.sel, self.irec )
        abr = "A %s " % self.a.brief 
        bbr = "B %s " % self.b.brief 
        return "\n".join([abn, abr, bbr])

    def __str__(self):
        lmx = self.ok.lmx
        if len(self.his.lines) > lmx:
            self.his.sli = slice(0,lmx)
        if len(self.mat.lines) > lmx:
            self.mat.sli = slice(0,lmx)
        pass
        return "\n".join(map(repr, [self,self.his,self.flg,self.mat]))
 
    def compare(self):
        log.debug("AB.compare START ")

        self.ahis = self._get_cf("all_seqhis_ana")
        self.amat = self._get_cf("all_seqmat_ana")

        if self.ok.prohis:self.prohis()
        if self.ok.promat:self.promat()
        log.debug("AB.compare DONE")


    def init_point(self):
        log.info("AB.init_point START")
        self.point = self.make_point()
        log.info("AB.init_point DONE")

    def point_dtype(self):
        dtype=[
                ('irl', int), 
                ('isel', int), 
                ('irec', int), 
                ('nrec', int), 
                ('reclab', "|S64")
              ]
        return dtype

    def _get_recpoint(self):
        """
        Returns None when single line selection is not active
        otherwise returns selected recpoint

        ::

            In [2]: ab.sel = "TO BT BT RE BT BT [SA]"

            In [3]: ab.recpoint
            Out[3]: (85, 12, 6, 7, 'TO BT BT RE BT BT [SA]')

            In [4]: ab.recpoint.isel
            Out[4]: 12

            In [5]: ab.recpoint.nrec
            Out[5]: 7

            In [6]: ab.recpoint.irec
            Out[6]: 6

        """
        rl = self.reclab
        if rl is None:
            return None
        pass 
        rp = self.point[self.point.reclab == rl]
        assert len(rp) == 1, rp
        return rp[0]
    recpoint = property(_get_recpoint)

    def make_point(self):
        """
        :return point: recarray for holding point level metadata

        ::

            In [5]: print "\n".join(map(repr, ab.point[:8]))
            (0, 0, 0, '[TO] BT BT BT BT SA')
            (1, 0, 1, 'TO [BT] BT BT BT SA')
            (2, 0, 2, 'TO BT [BT] BT BT SA')
            (3, 0, 3, 'TO BT BT [BT] BT SA')
            (4, 0, 4, 'TO BT BT BT [BT] SA')
            (5, 0, 5, 'TO BT BT BT BT [SA]')
            (6, 1, 0, '[TO] AB')
            (7, 1, 1, 'TO [AB]')

            In [6]: print "\n".join(map(repr, ab.point[-8:]))
            (64257, 4847, 8, 'TO RE RE RE RE BT BT BT [SC] BR BR BR BR BR BR BR')
            (64258, 4847, 9, 'TO RE RE RE RE BT BT BT SC [BR] BR BR BR BR BR BR')
            (64259, 4847, 10, 'TO RE RE RE RE BT BT BT SC BR [BR] BR BR BR BR BR')
            (64260, 4847, 11, 'TO RE RE RE RE BT BT BT SC BR BR [BR] BR BR BR BR')
            (64261, 4847, 12, 'TO RE RE RE RE BT BT BT SC BR BR BR [BR] BR BR BR')
            (64262, 4847, 13, 'TO RE RE RE RE BT BT BT SC BR BR BR BR [BR] BR BR')
            (64263, 4847, 14, 'TO RE RE RE RE BT BT BT SC BR BR BR BR BR [BR] BR')
            (64264, 4847, 15, 'TO RE RE RE RE BT BT BT SC BR BR BR BR BR BR [BR]')

        """ 
        rls = self.reclabs(0,None)
        nrls = len(rls)

        point = np.recarray((nrls,), dtype=self.point_dtype())
        point.reclab = rls
        point.irl = np.arange(0, nrls)

        offset = 0 
        for isel,clab in enumerate(self.clabels[0:None]):
            sls = Ctx.reclabs_(clab)
            nsls = len(sls)
            point.isel[offset:offset+nsls] = np.repeat( isel, nsls)
            point.irec[offset:offset+nsls] = np.arange( 0   , nsls)
            point.nrec[offset:offset+nsls] = np.repeat( nsls, nsls)
            offset += nsls 
        pass
        return point



    def prohis(self, rng=range(1,8)):
        for imsk in rng:
            setattr(self, "his_%d" % imsk, self.cf("seqhis_ana_%d" % imsk)) 
        pass
    def promat(self, rng=range(1,8)):
        for imsk in rng:
            setattr(self, "mat_%d" % imsk, self.cf("seqmat_ana_%d" % imsk)) 
        pass

    def tabname(self, ana):
        return ana.replace("_ana", "_tab")

    def _make_cf(self, ana="seqhis_ana"):
        """
        all_ tables have no selection applied so they are not dirtied by changing selection
        """
        c_tab = Evt.compare_ana( self.a, self.b, ana, lmx=self.ok.lmx, cmx=self.ok.cmx, c2max=None, cf=True)
        if not ana[0:3] == "all":
            self.tabs.append(c_tab)
        pass 
        tabname = self.tabname(ana)
        setattr(self, tabname, c_tab)
        return c_tab 

    def _set_dirty(self, dirty):
        for tab in self.tabs:
            tab.dirty = dirty
        pass
    def _get_dirty(self):
        dtabs = filter(lambda tab:tab.dirty, self.tabs)
        return len(dtabs) > 0 
    dirty = property(_get_dirty, _set_dirty)


    def _get_cf(self, ana):
        """
        Changing *sel* property invokes _set_sel 
        results in a change to the SeqAna in the A B Evt,
        thus all AB comparison tables are marked dirty, causing 
        them to be recreated at next access.
        """
        tabname = self.tabname(ana)
        tab = getattr(self, tabname, None)
        if tab is None:
            tab = self._make_cf(ana) 
        elif tab.dirty:
            tab = self._make_cf(ana) 
        else:
            pass 
        return tab

    def _get_his(self):
        return self._get_cf("seqhis_ana")
    def _get_mat(self):
        return self._get_cf("seqmat_ana")
    def _get_flg(self):
        return self._get_cf("pflags_ana")

    his = property(_get_his)
    mat = property(_get_mat)
    flg = property(_get_flg)

    # high level *sel* selection only, for lower level *psel* selections 
    # apply individually to evt a and b 

    def _get_sel(self):
        return self._sel
    def _set_sel(self, sel, nom="sel"):
        """
        NB slice selection at AB level must be
        converted into seq string selection at evt level
        as the labels can diverge, expecially out in the tail

        slice selection forces into seqhis selection
        """
        log.debug("AB._set_sel %s " % repr(sel))

        if type(sel) is slice:
            clabels = self.clabels
            seqs = clabels[sel]  
            assert len(seqs) == 1, seqs
            log.debug("AB._set_set convert slice selection %r into common label seq selection  %s " % (sel, seqs[0]))
            sel = seqs[0]
            self.flv = "seqhis"
        pass

        if nom == "sel":
            self.a.sel = sel
            self.b.sel = sel
        elif nom == "selhis":
            self.a.selhis = sel
            self.b.selhis = sel
        elif nom == "selmat":
            self.a.selmat = sel
            self.b.selmat = sel
        elif nom == "selflg":
            self.a.selflg = sel
            self.b.selflg = sel
        else:
            assert 0, nom 
        pass

        self._sel = sel 
        self.dirty = True  

    sel = property(_get_sel, _set_sel)
    
    def _set_selmat(self, sel):
        self._set_sel( sel, nom="selmat")
    def _set_selhis(self, sel):
        self._set_sel( sel, nom="selhis")
    def _set_selflg(self, sel):
        self._set_sel( sel, nom="selflg")

    selmat = property(_get_sel, _set_selmat)
    selhis = property(_get_sel, _set_selhis)
    selflg = property(_get_sel, _set_selflg)


    def _set_flv(self, flv):
        self.a.flv = flv
        self.b.flv = flv
    def _get_flv(self):
        a_flv = self.a.flv    
        b_flv = self.b.flv    
        assert a_flv == b_flv
        return a_flv
    flv = property(_get_flv, _set_flv)

    def _set_irec(self, irec):
        self.a.irec = irec 
        self.b.irec = irec 
    def _get_irec(self):
        a_irec = self.a.irec
        b_irec = self.b.irec
        assert a_irec == b_irec
        return a_irec
    irec = property(_get_irec, _set_irec)

    def _get_reclab(self):
        a_reclab = self.a.reclab
        b_reclab = self.b.reclab
        assert a_reclab == b_reclab
        return a_reclab
    reclab = property(_get_reclab)

    def _get_label0(self):
        a_label0 = self.a.label0
        b_label0 = self.b.label0
        assert a_label0 == b_label0
        return a_label0
    label0 = property(_get_label0)

    def _get_seq0(self):
        lab0 = self.label0
        if lab0 is None:
            return None
        return lab0.replace(" ","_")
    seq0 = property(_get_seq0)

    def count(self, line=0, col=1):
        """standard selects have only one sequence line""" 
        cu = self.seq.cu 
        if cu is None:
            return None
        pass
        return cu[line,col]

    def _get_aseq(self):
        """
        aseq is not changed by the current selection 
        """
        flv = self.flv
        if flv == "seqhis":
            return self.ahis
        elif flv == "seqmat":
            return self.amat
        else:
            pass
        return None
    aseq = property(_get_aseq)

    def _get_seq(self):
        """
        seq is changed by current selection
        """
        flv = self.flv
        if flv == "seqhis":
            return self.his
        elif flv == "seqmat":
            return self.mat
        else:
            pass
        return None
    seq = property(_get_seq)

    def _get_achi2(self):
        aseq = self.aseq 
        assert not aseq is None
        return aseq.c2
    achi2 = property(_get_achi2)

    def a_count(self, line=0):
        return self.count(line,1)

    def b_count(self, line=0):
        return self.count(line,2)

    def ab_count(self, line=0):
        ac = self.a_count(line)
        bc = self.b_count(line)
        return "%d/%d" % (ac,bc)

    def _get_suptitle(self):
        abc = self.ab_count()
        title = "%s/%s/%s : %s  :  %s " % (self.ok.tag, self.ok.det, self.ok.src, abc, self.reclab )
        return title
    suptitle = property(_get_suptitle)

    def _get_ctx(self):
        """
        hmm at some point seq0 lost its underscores
        """
        return Ctx({'det':self.ok.det, 'tag':self.ok.tag, 'src':self.ok.src, 'seq0':self.label0, 'lab':self.reclab, 'irec':self.irec, 'qwn':self.qwn })
    ctx = property(_get_ctx)

    pctx = property(lambda self:self.ctx.pctx)
    qctx = property(lambda self:self.ctx.qctx)


    def _get_nrec(self):
        """
        :return: number of record points, when a single seq selection is active
        """ 
        a_nr = self.a.nrec
        b_nr = self.b.nrec
        assert a_nr == b_nr, (a_nr, b_nr, "A and B should have same nrec ?")
        if a_nr == b_nr and a_nr != -1:
            nr = a_nr
        else:
            nr = -1
        pass
        log.debug(" a_nr:%d b_nr:%d nr:%d " % (a_nr, b_nr, nr) )
        return nr 
    nrec = property(_get_nrec)


    def iflg(self, flg):
        """
        :return: number of record points, when a single seq selection is active
        """ 
        a_ifl = self.a.iflg(flg)
        b_ifl = self.b.iflg(flg)
        assert a_ifl == b_ifl, (a_ifl, b_ifl, "A and B should have same iflg ?")
        if a_ifl == b_ifl and a_ifl != None:
            ifl = a_ifl
        else:
            ifl = None
        pass
        log.info(" a_ifl:%s b_ifl:%s ifl:%s " % (a_ifl, b_ifl, ifl) )
        return ifl 

    def nrecs(self, start=0, stop=None, step=1):
        """
        """
        sli = slice(start, stop, step)
        labels = self.clabels[sli] 
        nrs = np.zeros(len(labels), dtype=np.int32) 
        for ilab, lab in enumerate(labels):
            nrs[ilab] = len(lab.split())
        pass
        return nrs

    def stats_dtype(self, qwns=None):
        if qwns is None:
            qwns = self.ok.qwns
        pass
        dtype = [(i,np.int32) for i in "iv is na nb".split()] 
        dtype += [(s,"|S64") for s in "reclab".split()]
        dtype += [(q,np.float32) for q in list(qwns.replace(",",""))]
        dtype += [(q,np.float32) for q in "seqc2 distc2".split()]
        return dtype


    def reclabs(self, start=0, stop=5):
        """
        :param start: common label slice starting index
        :param stop: common label slice stop index, can be None
        :return rls: list of reclabels

        Full reclabs is very large list::

            In [5]: rl = ab.reclabs(0,None)

            In [6]: len(rl)
            Out[6]: 64265

            In [7]: len(ab.clabels)
            Out[7]: 4848

            In [8]: 64265./4848.        ## average seq0 length is 13 points 
            Out[8]: 13.255981848184819

            In [9]: rl[:5]
            Out[9]: 
            ['[TO] BT BT BT BT SA',
             'TO [BT] BT BT BT SA',
             'TO BT [BT] BT BT SA',
             'TO BT BT [BT] BT SA',
             'TO BT BT BT [BT] SA']

        """
        l = []
        for clab in self.clabels[start:stop]:
            l.extend(Ctx.reclabs_(clab))
        pass
        return l


    def stats_(self, reclabs, qwns=None, rehist=False, c2shape=True):
        """
        ::

            st = ab.stats_(ab.point.reclab[:10])

        """
        if qwns is None:
            qwns = self.ok.qwns
        pass

        nrl = len(reclabs)
        stat = np.recarray((nrl,), dtype=self.stats_dtype(qwns))

        for ival, rl in enumerate(reclabs):

            self.sel = rl     ## setting single line selection

            rp = self.recpoint

            isel = rp.isel

            # NB OrderedDict setting order must match dtype name order
            od = odict()
            od["iv"] = rp.irl
            od["is"] = isel
            od["na"] = self.aseq.cu[isel,1] 
            od["nb"] = self.aseq.cu[isel,2]
            od["reclab"] = rl

            hh = self.rhist(rehist=rehist, c2shape=c2shape)
            qd = odict(map(lambda h:(h.qwn,h.c2p), hh))
            od.update(qd)

            od["seqc2"] = self.seq.c2[0]
            od["distc2"] = CFH.c2per_(hh)

            stat[ival] = tuple(od.values())
        pass

        st = ABStat(stat) 
        st.save()
        return st 

 
    def stats(self, start=0, stop=5, qwns=None, rehist=False, c2shape=True):
        """
        slice selection in AB has to be converted into common seq selection 
        in the evt 
        """
        if qwns is None:
            qwns = self.ok.qwns
        pass
        qwns = qwns.replace(",","")

        dtype = self.stats_dtype(qwns)

        nrs = self.nrecs(start, stop)
        trs = nrs.sum()

        log.info("AB.stats : %s :  trs %d nrs %s " % ( qwns, trs, repr(nrs)) )
        
        stat = np.recarray((trs,), dtype=dtype)
        ival = 0
        for i,isel in enumerate(range(start, stop)):

            self.sel = slice(isel, isel+1)    ## changing single seq line selection 

            nr = self.nrec
            assert nrs[i] == nr, (i, nrs[i], nr )  

            for irec in range(nr):

                self.irec = irec 

                self.qwn = qwns   # for collective qctx 

                qctx = self.qctx

                key = self.suptitle

                log.debug("stats irec %d nrec %d ival %d key %s qctx %s " % (irec, nr, ival, key, qctx))


                # NB OrderedDict setting order must match dtype name order
                od = odict()
                od["iv"] = ival
                od["is"] = isel
                od["na"] = self.a_count(0)    # single line selection so line=0
                od["nb"] = self.b_count(0)    # single line selection so line=0
                od["reclab"] = self.reclab 

                hh = self.rhist(qwns, irec, rehist=rehist, c2shape=c2shape)
                qd = odict(map(lambda h:(h.qwn,h.c2p), hh))
                od.update(qd)

                od["seqc2"] = self.seq.c2[0]
                od["distc2"] = CFH.c2per_(hh)

                stat[ival] = tuple(od.values())
                ival += 1
            pass
        pass
        log.info("AB.stats histogramming done")
        assert ival == trs, (ival, trs )

        st = ABStat(self.ok, stat) 
        st.save()
        return st 

    def totrec(self, start=0, stop=None, step=1):
        """
        :param sli:

        ::

            In [2]: ab.totrec()    # union of labels brings in a lot more of them
            Out[2]: 64265

            In [3]: ab.a.totrec()
            Out[3]: 43177

            In [4]: ab.b.totrec()
            Out[4]: 43085

        """
        nrs = self.nrecs(start, stop, step)
        return int(nrs.sum())

    def _get_clabels(self):
        """
        :return clabels: all labels of current flv 
        """
        flv = self.flv
        if flv == "seqhis":
            clabels = self.ahis.labels
        elif flv == "seqmat":
            clabels = self.amat.labels
        else:
            clabels = []
        pass
        return clabels
    clabels = property(_get_clabels)

    def rpost(self):
        """
        Sliced decompression within a selection 
        works via the rx replacement with rx[psel] 
        done in Evt.init_selection.
        """
        self.checkrec()
        aval = self.a.rpost()
        bval = self.b.rpost()
        return aval, bval


    def checkrec(self,fr=0,to=0):
        nr = self.nrec
        if nr > -1 and fr < nr and to < nr:
            return True 
        log.fatal("checkrec requires a single label selection nr %d fr %d to %d" % (nr,fr,to))
        return False

    def rdir(self, fr=0, to=1):
        if not self.checkrec(fr,to):
            return None
        aval = self.a.rdir(fr,to)
        bval = self.b.rdir(fr,to)
        return aval, bval

    def rpol_(self, fr):
        if not self.checkrec(fr):
            return None
        aval = self.a.rpol_(fr)
        bval = self.b.rpol_(fr)
        return aval, bval

    def rpol(self):
        if not self.checkrec():
            return None
        aval = self.a.rpol()
        bval = self.b.rpol()
        return aval, bval

    def rw(self):
        if not self.checkrec():
            return None
        aval = self.a.rw()
        bval = self.b.rw()
        return aval, bval


    @classmethod
    def rrandhist(cls):
        bn = np.linspace(-4,4,200)
        av = np.random.standard_normal(8000)
        bv = np.random.standard_normal(8000)
        la = ["A rand", "B rand"]

        ctx = {}
        ctx["det"] = "dummy"
        ctx["tag"] = "1"
        ctx["seq"] = "dummy"
        ctx["qwn"] = "U"
        ctx["irec"] = "0"   

        cfh = CFH(ctx)
        cfh(bn,av,bv,la,c2cut=cls.C2CUT)
        return cfh
 
    def _set_qwn(self, qwn):
        self._qwn = qwn 
    def _get_qwn(self):
        return self._qwn
    qwn = property(_get_qwn, _set_qwn)

    def _get_h(self):
        hh = self.rhist( self.qwn, self.irec)
        assert len(hh) == 1
        return hh[0]        
    h = property(_get_h)

    def _get_hh(self):
        hh = self.rhist()
        return hh
    hh = property(_get_hh)


    def rhist_(self, ctxs, rehist=False, c2shape=False):
        """
        :param ctxs: list of Ctx 
        :param rehist: bool, when True recreate histograms, otherwise pre-persisted histos are loaded


        Collects unique seq0 and then interactes 

        """
        seq0s = list(set(map(lambda ctx:ctx.seq0,ctxs)))
        hh = []
        for seq0 in seq0s:
            sel = seq0.replace("_", " ")
            log.info("AB.rhist_ setting sel to \"%s\" rehist %d  " % (sel, rehist) )

            self.sel = sel   # adjust selection 

            for ctx in filter(lambda ctx:ctx.seq0 == seq0, ctxs):
                hs = self.rhist(qwn=ctx["qwn"], irec=int(ctx["irec"]), rehist=rehist, c2shape=c2shape)
                assert len(hs) == 1
                hh.append(hs[0])
            pass
        pass
        return hh

    def rhist(self, qwn=None, irec=None, rehist=False, log_=False, c2shape=True ): 

        if qwn is None:
            qwn = self.ok.qwns
        assert type(qwn) is str

        if irec is None:
            irec = self.irec
        assert type(irec) is int

        hh = []

        srec = CFH.srec_(irec)

        log.debug("AB.rhist qwn %s irec %s srec %s " % (qwn, irec, srec))

        for ir in CFH.irec_(srec):

            log.debug(" ir %s ir 0x%x " % (ir,ir))

            self.irec = ir

            for q in str(qwn):

                self.qwn = q 

                ctx=self.ctx

                h = CFH(ctx)

                h.log = log_

                if h.exists() and not rehist:
                    h.load()
                else:
                    bn, av, bv, la = self.rqwn(q, ir)

                    h(bn,av,bv,la, c2cut=self.C2CUT, c2shape=c2shape )

                    h.save()
                pass
                hh.append(h)
            pass
        pass
        return hh
 
    def rqwn(self, qwn, irec): 
        """
        :param qwn: X,Y,Z,W,T,A,B,C or R  
        :param irec: step index 0,1,...
        :return binx, aval, bval, labels

        ::

            bi,a,b,l = cf.rqwn("T",4)

        """
        a = self.a
        b = self.b
        lval = "%s[%s]" % (qwn.lower(), irec)
        labels = ["Op : %s" % lval, "G4 : %s" % lval]
 
        if qwn == "R":
            apost = a.rpost_(irec)
            bpost = b.rpost_(irec)
            aval = vnorm(apost[:,:2])
            bval = vnorm(bpost[:,:2])
            cbins = a.pbins()
        elif qwn == "W":
            aval = a.recwavelength(irec)
            bval = b.recwavelength(irec)
            cbins = a.wbins()
        elif qwn in Evt.RPOST:
            q = Evt.RPOST[qwn]
            aval = a.rpost_(irec)[:,q]
            bval = b.rpost_(irec)[:,q]
            if qwn == "T":
                cbins = a.tbins()
            else:
                cbins = a.pbins()
            pass
        elif qwn in Evt.RPOL:
            q = Evt.RPOL[qwn]
            aval = a.rpol_(irec)[:,q]
            bval = b.rpol_(irec)[:,q]
            cbins = a.rpol_bins()
        else:
            assert 0, "qwn %s unknown " % qwn 
        pass

        binscale = Evt.RQWN_BINSCALE[qwn]
        bins = cbins[::binscale]   

        # hmm should arrange scaling to retain extreme bins to avoid clipping perhaps ??
        # formerly tried clever binning, but certainty of simple binning turns out easier to interpret
        #    bins = decompression_bins(cbins, [aval, bval], label=lval, binscale=binscale )
        #  

        if len(bins) == 0:
            raise Exception("no bins")

        return bins, aval, bval, labels



if __name__ == '__main__':
    ok = opticks_main()
    ab = AB(ok)
    print ab
    
