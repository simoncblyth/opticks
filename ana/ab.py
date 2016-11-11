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
    def __init__(self, args):
        self.args = args
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

    def _make_cf(self, ana="seqhis_ana"):
        c_tab = Evt.compare_ana( self.a, self.b, ana, lmx=self.args.lmx, cmx=self.args.cmx, c2max=None, cf=True)
        return c_tab 

    def _get_cf(self, ana):
        """
        Changing selection changes the SeqAna in the A B Evt, so 
        to follow that a dirty flag is set on changing selection
        to force recomparison.

        Hmm need separate dirty labels ?
        """
        if not hasattr(self, ana) or self._dirty == True:
            setattr(self, ana, self._make_cf(ana))
        return getattr(self, ana)

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
        log.info("AB._set_sel %s " % repr(sel))
        self.a.sel = sel
        self.b.sel = sel
        self._sel = sel 
        self._dirty = True  
    sel = property(_get_sel, _set_sel)


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


 
    def a_count(self, line=0):
        """subselects usually have only one sequence line""" 
        return self.his.cu[line,1]

    def b_count(self, line=0):
        return self.his.cu[line,2]

    def ab_count(self, line=0):
        ac = self.a_count(line)
        bc = self.b_count(line)
        return "%d/%d" % (ac,bc)

    def suptitle(self, irec=-1):
        abc = self.ab_count()
        lab = self.seqlab(irec)
        title = "%s/%s/%s : %s  :  %s " % (self.args.tag, self.args.det, self.args.src, abc, lab )
        return title

    def ctx(self, **kwa):
        irec = kwa.get('irec',"0")
        self.irec = irec
        d = {'det':self.args.det, 'tag':self.args.tag, 'src':self.args.src, 'seq':self.seqs[0].replace(" ","_"), 'lab':self.reclab }
        d.update(kwa)
        return d 

    def nrec(self):
        """
        :return: number of steps, when a single sequence is selected
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

    def rhist(self, qwn, irec, cut=30): 
        bn, av, bv, la = self.rqwn(qwn, irec)
        ctx=self.ctx(qwn=qwn,irec=irec)
        cfh = CFH(ctx)
        cfh(bn,av,bv,la,cut=cut)
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

