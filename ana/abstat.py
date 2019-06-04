#!/usr/bin/env python
"""

ABStat slicing::

    In [22]: st[10:20]
    Out[22]: 
    ABStat 10 slice(10, 20, None) na,nb,qctx,reclab,X,Y,Z,T,A,B,C,R 
    ===== ===== =============================== ====================== ===== ===== ===== ===== ===== ===== ===== ===== 
    na    nb    qctx                            reclab                 X     Y     Z     T     A     B     C     R     
    ===== ===== =============================== ====================== ===== ===== ===== ===== ===== ===== ===== ===== 
    45490 45054 TO_SC_BT_BT_BT_BT_SA/2/XYZTABCR TO SC [BT] BT BT BT SA  1.12  1.29  1.12  0.86  0.70  1.03  1.00  0.89 
    45490 45054 TO_SC_BT_BT_BT_BT_SA/3/XYZTABCR TO SC BT [BT] BT BT SA  1.03  1.39  1.17  0.87  0.71  0.98  1.01  0.76 
    45490 45054 TO_SC_BT_BT_BT_BT_SA/4/XYZTABCR TO SC BT BT [BT] BT SA  1.04  0.93  0.85  1.63  0.70  0.99  1.01  1.11 
    45490 45054 TO_SC_BT_BT_BT_BT_SA/5/XYZTABCR TO SC BT BT BT [BT] SA  0.98  0.91  0.82  1.62  1.02  1.01  0.99  1.21 
    45490 45054 TO_SC_BT_BT_BT_BT_SA/6/XYZTABCR TO SC BT BT BT BT [SA]  0.80  0.99  1.01  4.78  1.02  1.01  0.99  1.18 
    28955 28649 TO_BT_BT_BT_BT_AB/0/XYZTABCR    [TO] BT BT BT BT AB     1.63  1.63  1.63  1.63  1.63  1.63  1.63  1.63 
    28955 28649 TO_BT_BT_BT_BT_AB/1/XYZTABCR    TO [BT] BT BT BT AB     1.63  1.63  1.63  1.63  1.63  1.63  1.63  1.63 
    28955 28649 TO_BT_BT_BT_BT_AB/2/XYZTABCR    TO BT [BT] BT BT AB     1.63  1.63  1.63  1.63  1.63  1.63  1.63  1.63 
    28955 28649 TO_BT_BT_BT_BT_AB/3/XYZTABCR    TO BT BT [BT] BT AB     1.63  1.63  1.63  1.63  1.63  1.63  1.63  1.63 
    28955 28649 TO_BT_BT_BT_BT_AB/4/XYZTABCR    TO BT BT BT [BT] AB     1.63  1.63  1.63  1.63  1.63  1.63  1.63  1.63 
    ===== ===== =============================== ====================== ===== ===== ===== ===== ===== ===== ===== ===== 

    ABStat 10 slice(10, 20, None) na,nb,qctx,reclab,X,Y,Z,T,A,B,C,R 


Lookup a line via reclab::

    In [10]: st[st.st.reclab == "TO BT BT BT BT DR BT BT BT BT BT BT BT BT [SA]"]
    Out[10]: 
    ABStat 1 iv,is,na,nb,qctx,reclab,X,Y,Z,T,A,B,C,R 
    == == ==== ==== ======================================================= ============================================== ===== ===== ===== ===== ===== ===== ===== ===== 
    iv is na   nb   qctx                                                    reclab                                         X     Y     Z     T     A     B     C     R     
    == == ==== ==== ======================================================= ============================================== ===== ===== ===== ===== ===== ===== ===== ===== 
    78 11 5339 5367 TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_BT_BT_BT_BT_SA/e/XYZTABCR TO BT BT BT BT DR BT BT BT BT BT BT BT BT [SA]  1.05  1.24  1.00 79.14  0.96  1.28  1.14  1.00 
    == == ==== ==== ======================================================= ============================================== ===== ===== ===== ===== ===== ===== ===== ===== 


"""
import os, logging, numpy as np
import numpy.lib.recfunctions as rf

from opticks.ana.make_rst_table import recarray_as_rst

log = logging.getLogger(__name__)


class ABStat(object):
    """
    Hmm stats should probably have a standard path within the event tree 
    """
    STATPATH = "$TMP/stat.npy"
    SUPTITLE = "  %(det)s/%(src)s/%(tag)s  %(iv)s/%(is)s %(na)d/%(nb)d   %(reclab)-50s  XYZT: %(X)4.2f %(Y)4.2f %(Z)4.2f %(T)4.2f ABCW: %(A)4.2f %(B)4.2f %(C)4.2f %(W)4.2f  seqc2 %(seqc2)4.2f dstc2 %(distc2)4.2f " 
    SKIP = "qctx".split()

    @classmethod
    def path_(cls):
        return os.path.expandvars(cls.STATPATH)

    def __init__(self, ok, st):
        self.ok = ok 
        self.st = st
        self.ar = self.ary
        self.sli = slice(0,None,1)

    def save(self):
        np.save(self.path_(),self.st)  

    @classmethod
    def load(cls, ok):
        ra = np.load(cls.path_()).view(np.recarray)
        return cls(ok, ra) 

    def __repr__(self):
        return "\n".join([self.brief,recarray_as_rst(self.st[self.sli], skip=self.SKIP),self.brief])

    def _get_brief(self):
        return "ABStat %s %s " % (len(self.st[self.sli]), ",".join(self.names) )
    brief = property(_get_brief)

    def _get_names(self):
        names = map( lambda k:None if k in self.SKIP else k, self.st.dtype.names )
        return filter(None, names)
    names = property(_get_names)

    def __getitem__(self, sli):
        if type(sli) is int:
            sli = slice(sli,sli+1)
        pass
        self.sli = sli
        return self

    def _get_ary(self, qwns=None):
        if qwns is None:
            qwns = list(self.ok.qwns)
        pass
        return np.vstack([self.st[q] for q in qwns]).T
    ary = property(_get_ary) 

    def _get_suptitle(self):
        """
        After making a single line selection this provides a plot title::

            In [11]: st[26].suptitle
            Out[11]: ' 26/5 20238/20140   TO [RE] BT BT BT BT SA                              XYZT: 0.85 0.00 0.00 1.31 ABCW: 1.12 1.37 1.10 0.78  seqc2 0.24 distc2 1.10 '

            In [16]: st[st.st.reclab=="TO RE BT [BT] BT BT DR BT BT BT BT SA"]
            Out[16]: 
            ABStat 1 iv,is,na,nb,reclab,X,Y,Z,T,A,B,C,W,seqc2,distc2 
            === == === === ===================================== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====== 
            iv  is na  nb  reclab                                X     Y     Z     T     A     B     C     W     seqc2 distc2 
            === == === === ===================================== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====== 
            868 97 125 126 TO RE BT [BT] BT BT DR BT BT BT BT SA  0.00  0.00  0.00  1.84  0.00  0.00  0.00  1.24  0.00  1.08  
            === == === === ===================================== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====== 

            ABStat 1 iv,is,na,nb,reclab,X,Y,Z,T,A,B,C,W,seqc2,distc2 

            In [17]: st.suptitle
            Out[17]: ' 868/97 125/126   TO RE BT [BT] BT BT DR BT BT BT BT SA               XYZT: 0.00 0.00 0.00 1.84 ABCW: 0.00 0.00 0.00 1.24  seqc2 0.00 distc2 1.08 '

        """
        nli = len(self.st[self.sli]) 
        if nli == 1:
            st_tup = self.st[self.sli][0] 

            d = {}
            d.update(self.ok.ctx)
            d.update(dict(zip(self.st.dtype.names,st_tup)))
            return self.SUPTITLE % d
        pass
        log.warning("st.suptitle requires single line slice")
        return None
    suptitle = property(_get_suptitle)


    def chi2sel(self, chi2cut=None,  statcut=None, style=None):
        """
        :return indices: propagation record point indices with chi2 sum greater than cut
        """
        if chi2cut is None:
            chi2cut = self.ok.chi2selcut
        if statcut is None:
            statcut = self.ok.statcut
        if style is None:
            style = "distc2stat"
        pass

        if style == "distc2":
            s = np.where( self.st.distc2 > chi2cut) 
        elif style == "distc2stat":
            s = np.where( np.logical_and(self.st.distc2 > chi2cut, self.st.na > statcut)) 
        elif style == "seqc2":
            s = np.where( self.st.seqc2 > chi2cut) 
        elif style == "qwnsum":
            s = np.where( np.sum(self.ar, axis=1) > chi2cut )
        else:
            assert 0, style
        pass

        log.info("style %s chi2cut %s statcut %s nsel %d " % (style, chi2cut, statcut, len(s[0]))) 

        return s 

    def qctxsel(self, chi2cut=None, statcut=None, style="qwnsum"):
        """
        :return qctx list: were chi2 sum exceeds the cut 
        """ 
        return self.st.qctx[self.chi2sel(chi2cut=chi2cut, statcut=statcut, style=style)]

    def reclabsel(self, chi2cut=None, statcut=None, style="distc2stat"):
        """
        :return reclab list: were chi2 sum exceeds the cut 
        """ 
        return self.st.reclab[self.chi2sel(chi2cut=chi2cut, statcut=statcut, style=style)]


    @classmethod
    def dump(cls, st=None): 
        if st is None:
            st = cls.load()
        print st


if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    ok = opticks_main()
    st = ABStat.load(ok)
    
    #print st 
    print st[st.chi2sel()]

    ar = st.ar

          
