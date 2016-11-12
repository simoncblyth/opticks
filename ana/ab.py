#!usr/bin/env python
"""

"""
import os, sys, logging, numpy as np

from opticks.ana.base import opticks_main
from opticks.ana.cfh import CFH
from opticks.ana.nbase import chi2, vnorm
from opticks.ana.decompression import decompression_bins
from opticks.ana.histype import HisType
from opticks.ana.mattype import MatType
from opticks.ana.evt import Evt

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

    """
    def __init__(self, args):
        self.args = args
        self.tabs = []
        self.load()
        self.compare()

    def load(self):
        """
        It takes aound 6s to load 1M full AB evt pair. So avoid needing to duplicate that.
        """
        log.info("AB.load START ")
        args = self.args
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
        abn = "AB(%s,%s,%s)  %s %s " % (self.args.tag, self.args.src, self.args.det, self.sel, self.irec )
        abr = "A %s " % self.a.brief 
        bbr = "B %s " % self.b.brief 
        return "\n".join([abn, abr, bbr])

    def __str__(self):
        lmx = self.args.lmx
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

        if self.args.prohis:self.prohis()
        if self.args.promat:self.promat()
        log.debug("AB.compare DONE")

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
        c_tab = Evt.compare_ana( self.a, self.b, ana, lmx=self.args.lmx, cmx=self.args.cmx, c2max=None, cf=True)
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
    def _set_sel(self, sel):
        log.debug("AB._set_sel %s " % repr(sel))
        self.a.sel = sel
        self.b.sel = sel
        self._sel = sel 
        self.dirty = True  
    sel = property(_get_sel, _set_sel)

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
        b_reclab = self.a.reclab
        assert a_reclab == b_reclab
        return a_reclab
    reclab = property(_get_reclab)

    def _get_label0(self):
        a_label0 = self.a.label0
        b_label0 = self.a.label0
        assert a_label0 == b_label0
        return a_label0
    label0 = property(_get_label0)

    def _get_seq0(self):
        lab0 = self.label0
        if lab0 is None:
            return None
        return lab0.replace(" ","_")
    seq0 = property(_get_seq0)

    def a_count(self, line=0):
        """subselects usually have only one sequence line""" 
        return self.his.cu[line,1]

    def b_count(self, line=0):
        return self.his.cu[line,2]

    def ab_count(self, line=0):
        ac = self.a_count(line)
        bc = self.b_count(line)
        return "%d/%d" % (ac,bc)

    def _get_suptitle(self):
        abc = self.ab_count()
        title = "%s/%s/%s : %s  :  %s " % (self.args.tag, self.args.det, self.args.src, abc, self.reclab )
        return title
    suptitle = property(_get_suptitle)


    def ctx(self, **kwa):
        irec = kwa.get('irec',"0")
        self.irec = irec
        d = {'det':self.args.det, 'tag':self.args.tag, 'src':self.args.src, 'seq0':self.seq0, 'lab':self.reclab }
        d.update(kwa)
        return d 

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
        log.info(" a_nr:%d b_nr:%d nr:%d " % (a_nr, b_nr, nr) )
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
        labels = self.alabels[sli] 
        nrs = np.zeros(len(labels), dtype=np.int32) 
        for ilab, lab in enumerate(labels):
            nrs[ilab] = len(lab.split())
        pass
        return nrs

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

    def _get_alabels(self):
        """
        :return alabels: all labels of current flv 
        """
        flv = self.flv
        if flv == "seqhis":
            alabels = self.ahis.labels
        elif flv == "seqmat":
            alabels = self.amat.labels
        else:
            alabels = []
        pass
        return alabels
    alabels = property(_get_alabels)

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
        cfh(bn,av,bv,la,cut=30)
        return cfh
 
    def _set_qwn(self, qwn):
        self._qwn = qwn 
    def _get_qwn(self):
        return self._qwn
    qwn = property(_get_qwn, _set_qwn)

    def _get_h(self):
        cfh = self.rhist( self.qwn, self.irec)
        return cfh        
    h = property(_get_h)

    def rhist(self, qwn, irec, cut=30): 
        ctx=self.ctx(qwn=qwn,irec=irec)
        cfh = CFH(ctx)
        if cfh.exists():
            cfh.load()
        else:
            bn, av, bv, la = self.rqwn(qwn, irec)
            cfh(bn,av,bv,la,cut=cut)
            cfh.save()
        pass
        return cfh
 
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

        if qwn in "ABC":
            # polarization is char compressed so have to use primordial bins
            bins = cbins
        else: 
            bins = decompression_bins(cbins, [aval, bval], label=lval, binscale=Evt.RQWN_BINSCALE[qwn] )


        if len(bins) == 0:
            raise Exception("no bins")

        return bins, aval, bval, labels



if __name__ == '__main__':
    ok = opticks_main()
    ab = AB(ok)
    print ab

