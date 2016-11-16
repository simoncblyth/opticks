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


Corresponding ndarray gives raw chi2 access::

    In [23]: a[10:20]
    Out[23]: 
    array([[ 1.1243,  1.285 ,  1.1159,  0.8648,  0.7049,  1.0259,  0.9997,  0.8856],
           [ 1.032 ,  1.3914,  1.1746,  0.875 ,  0.7139,  0.9835,  1.0135,  0.7598],
           [ 1.0422,  0.9272,  0.8455,  1.6319,  0.6992,  0.9856,  1.0088,  1.1129],
           [ 0.9838,  0.9115,  0.8178,  1.6159,  1.017 ,  1.0052,  0.9891,  1.2069],
           [ 0.7997,  0.9926,  1.0145,  4.7815,  1.017 ,  1.0052,  0.9891,  1.1766],
           [ 1.6255,  1.6255,  1.6255,  1.6255,  1.6255,  1.6255,  1.6255,  1.6255],
           [ 1.6255,  1.6255,  1.6255,  1.6255,  1.6255,  1.6255,  1.6255,  1.6255],
           [ 1.6255,  1.6255,  1.6255,  1.6255,  1.6255,  1.6255,  1.6255,  1.6255],
           [ 1.6255,  1.6255,  1.6255,  1.6255,  1.6255,  1.6255,  1.6255,  1.6255],
           [ 1.6255,  1.6255,  1.6255,  1.6255,  1.6255,  1.6255,  1.6255,  1.6255]], dtype=float32)


    In [8]: st[np.where( np.sum(ar, axis=1) > 30 )]    # propagation record points with chi2 sum greater than 30 
    === == ===== ===== ========================================================== ================================================= ===== ===== ===== ====== ===== ===== ===== ===== 
    iv  is na    nb    qctx                                                       reclab                                            X     Y     Z     T      A     B     C     R     
    === == ===== ===== ========================================================== ================================================= ===== ===== ===== ====== ===== ===== ===== ===== 
    20  3  28955 28649 TO_BT_BT_BT_BT_AB/5/XYZTABCR                               TO BT BT BT BT [AB]                                1.81  1.63  1.63 23.31   1.63  1.63  1.63  1.81 
    70  11 5339  5367  TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_BT_BT_BT_BT_SA/6/XYZTABCR    TO BT BT BT BT DR [BT] BT BT BT BT BT BT BT SA     0.28  0.91  0.36 279.80  1.27  1.20  0.97  0.15 
    71  11 5339  5367  TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_BT_BT_BT_BT_SA/7/XYZTABCR    TO BT BT BT BT DR BT [BT] BT BT BT BT BT BT SA     0.27  1.04  0.64 265.52  1.19  1.16  1.08  0.07 
    72  11 5339  5367  TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_BT_BT_BT_BT_SA/8/XYZTABCR    TO BT BT BT BT DR BT BT [BT] BT BT BT BT BT SA     1.87  0.99  0.45 106.27  1.08  1.07  1.01  0.67 
    73  11 5339  5367  TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_BT_BT_BT_BT_SA/9/XYZTABCR    TO BT BT BT BT DR BT BT BT [BT] BT BT BT BT SA     1.51  1.24  0.28 44.65   1.16  1.16  1.08  0.55 
    79  12 5113  4868  TO_BT_BT_RE_BT_BT_SA/0/XYZTABCR                            [TO] BT BT RE BT BT SA                             6.01  6.01  6.01  6.01   6.01  6.01  6.01  6.01 
    80  12 5113  4868  TO_BT_BT_RE_BT_BT_SA/1/XYZTABCR                            TO [BT] BT RE BT BT SA                             6.01  6.01  6.01  6.01   6.01  6.01  6.01  6.01 
    81  12 5113  4868  TO_BT_BT_RE_BT_BT_SA/2/XYZTABCR                            TO BT [BT] RE BT BT SA                             6.01  6.01  6.01  6.01   6.01  6.01  6.01  6.01 
    95  14 4494  4420  TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_SA/6/XYZTABCR                TO BT BT BT BT DR [BT] BT BT BT SA                 3.42  1.18  1.24 21.20   1.34  1.56  1.21  0.64 
    96  14 4494  4420  TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_SA/7/XYZTABCR                TO BT BT BT BT DR BT [BT] BT BT SA                 3.96  1.44  1.37 19.25   1.27  1.49  1.30  0.87 
    191 25 1260  1263  TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_AB/6/XYZTABCR                TO BT BT BT BT DR [BT] BT BT BT AB                 4.67  1.39  1.29 74.24   0.76  1.13  0.52  2.46 
    203 27 1104  1199  TO_BT_BT_RE_BT_BT_BT_BT_BT_BT_SA/0/XYZTABCR                [TO] BT BT RE BT BT BT BT BT BT SA                 3.92  3.92  3.92  3.92   3.92  3.92  3.92  3.92 
    204 27 1104  1199  TO_BT_BT_RE_BT_BT_BT_BT_BT_BT_SA/1/XYZTABCR                TO [BT] BT RE BT BT BT BT BT BT SA                 3.92  3.92  3.92  3.92   3.92  3.92  3.92  3.92 
    205 27 1104  1199  TO_BT_BT_RE_BT_BT_BT_BT_BT_BT_SA/2/XYZTABCR                TO BT [BT] RE BT BT BT BT BT BT SA                 3.92  3.92  3.92  3.92   3.92  3.92  3.92  3.92 
    213 27 1104  1199  TO_BT_BT_RE_BT_BT_BT_BT_BT_BT_SA/10/XYZTABCR               TO BT BT RE BT BT BT BT BT BT [SA]                 3.92  3.92  3.92  3.92   3.92  3.92  3.92  3.92 
    241 31 1067  1019  TO_BT_BT_BT_BT_DR_BT_BT_AB/6/XYZTABCR                      TO BT BT BT BT DR [BT] BT AB                       1.90  1.56  1.07 21.19   1.83  1.72  0.95  1.53 
    272 37 817   733   TO_SC_BT_BT_SC_BT_BT_SA/0/XYZTABCR                         [TO] SC BT BT SC BT BT SA                          4.55  4.55  4.55  4.55   4.55  4.55  4.55  4.55 
    313 42 545   566   TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_SC_BT_BT_BT_BT_SA/6/XYZTABCR TO BT BT BT BT DR [BT] BT BT BT SC BT BT BT BT SA  0.37  1.21  1.38 78.85   0.00  0.91  0.71  0.30 
    314 42 545   566   TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_SC_BT_BT_BT_BT_SA/7/XYZTABCR TO BT BT BT BT DR BT [BT] BT BT SC BT BT BT BT SA  0.34  1.08  1.34 78.18   0.29  0.88  0.59  0.30 
    323 43 538   460   TO_SC_BT_BT_BT_BT_DR_SA/0/XYZTABCR                         [TO] SC BT BT BT BT DR SA                          6.10  6.10  6.10  6.10   6.10  6.10  6.10  6.10 
    385 50 385   311   TO_RE_BT_BT_SC_BT_BT_SA/0/XYZTABCR                         [TO] RE BT BT SC BT BT SA                          7.87  7.87  7.87  7.87   7.87  7.87  7.87  7.87 
    527 66 285   239   TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_BT_BT_AB/0/XYZTABCR          [TO] BT BT BT BT DR BT BT BT BT BT BT AB           4.04  4.04  4.04  4.04   4.04  4.04  4.04  4.04 
    528 66 285   239   TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_BT_BT_AB/1/XYZTABCR          TO [BT] BT BT BT DR BT BT BT BT BT BT AB           4.04  4.04  4.04  4.04   4.04  4.04  4.04  4.04 
    529 66 285   239   TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_BT_BT_AB/2/XYZTABCR          TO BT [BT] BT BT DR BT BT BT BT BT BT AB           4.04  4.04  4.04  4.04   4.04  4.04  4.04  4.04 
    530 66 285   239   TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_BT_BT_AB/3/XYZTABCR          TO BT BT [BT] BT DR BT BT BT BT BT BT AB           4.04  4.04  4.04  4.04   4.04  4.04  4.04  4.04 
    531 66 285   239   TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_BT_BT_AB/4/XYZTABCR          TO BT BT BT [BT] DR BT BT BT BT BT BT AB           4.04  4.04  4.04  4.04   4.04  4.04  4.04  4.04 
    537 66 285   239   TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_BT_BT_AB/10/XYZTABCR         TO BT BT BT BT DR BT BT BT BT [BT] BT AB           4.04  4.04  4.04  4.04   4.04  4.04  4.04  4.04 
    538 66 285   239   TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_BT_BT_AB/11/XYZTABCR         TO BT BT BT BT DR BT BT BT BT BT [BT] AB           4.04  4.04  4.04  4.04   4.04  4.04  4.04  4.04 
    539 66 285   239   TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_BT_BT_AB/12/XYZTABCR         TO BT BT BT BT DR BT BT BT BT BT BT [AB]           4.04  4.04  4.04  4.04   4.04  4.04  4.04  4.04 
    617 74 164   201   TO_BT_BT_SC_BT_BT_BT_BT_BT_BT_AB/0/XYZTABCR                [TO] BT BT SC BT BT BT BT BT BT AB                 3.75  3.75  3.75  3.75   3.75  3.75  3.75  3.75 
    618 74 164   201   TO_BT_BT_SC_BT_BT_BT_BT_BT_BT_AB/1/XYZTABCR                TO [BT] BT SC BT BT BT BT BT BT AB                 3.75  3.75  3.75  3.75   3.75  3.75  3.75  3.75 
    619 74 164   201   TO_BT_BT_SC_BT_BT_BT_BT_BT_BT_AB/2/XYZTABCR                TO BT [BT] SC BT BT BT BT BT BT AB                 3.75  3.75  3.75  3.75   3.75  3.75  3.75  3.75 
    627 74 164   201   TO_BT_BT_SC_BT_BT_BT_BT_BT_BT_AB/10/XYZTABCR               TO BT BT SC BT BT BT BT BT BT [AB]                 3.75  3.75  3.75  3.75   3.75  3.75  3.75  3.75 
    756 88 136   103   TO_RE_BT_BT_RE_RE_BT_BT_SA/0/XYZTABCR                      [TO] RE BT BT RE RE BT BT SA                       4.56  4.56  4.56  4.56   4.56  4.56  4.56  4.56 
    777 90 98    131   TO_BT_BT_RE_BT_BT_SC_BT_BT_BT_BT_SA/0/XYZTABCR             [TO] BT BT RE BT BT SC BT BT BT BT SA              4.76  4.76  4.76  4.76   4.76  4.76  4.76  4.76 
    778 90 98    131   TO_BT_BT_RE_BT_BT_SC_BT_BT_BT_BT_SA/1/XYZTABCR             TO [BT] BT RE BT BT SC BT BT BT BT SA              4.76  4.76  4.76  4.76   4.76  4.76  4.76  4.76 
    779 90 98    131   TO_BT_BT_RE_BT_BT_SC_BT_BT_BT_BT_SA/2/XYZTABCR             TO BT [BT] RE BT BT SC BT BT BT BT SA              4.76  4.76  4.76  4.76   4.76  4.76  4.76  4.76 
    787 90 98    131   TO_BT_BT_RE_BT_BT_SC_BT_BT_BT_BT_SA/10/XYZTABCR            TO BT BT RE BT BT SC BT BT BT [BT] SA              4.76  4.76  4.76  4.76   4.76  4.76  4.76  4.76 
    788 90 98    131   TO_BT_BT_RE_BT_BT_SC_BT_BT_BT_BT_SA/11/XYZTABCR            TO BT BT RE BT BT SC BT BT BT BT [SA]              4.76  4.76  4.76  4.76   4.76  4.76  4.76  4.76 
    === == ===== ===== ========================================================== ================================================= ===== ===== ===== ====== ===== ===== ===== ===== 



"""
import os, logging, numpy as np
import numpy.lib.recfunctions as rf

from opticks.ana.base import opticks_main
from opticks.ana.make_rst_table import recarray_as_rst

log = logging.getLogger(__name__)


class ABStat(object):
    """
    """
    STATPATH = "$TMP/stat.npy"
    QWNS = "XYZTABCR"

    @classmethod
    def path_(cls):
        return os.path.expandvars(cls.STATPATH)

    def __init__(self, st):
        self.st = st
        self.ar = self.ary
        self.sli = slice(0,None,1)

    def save(self):
        np.save(self.path_(),self.st)  

    @classmethod
    def load(cls):
        ra = np.load(cls.path_()).view(np.recarray)
        return cls(ra) 

    def __repr__(self):
        return "\n".join([self.brief,recarray_as_rst(self.st[self.sli]),self.brief])

    def _get_brief(self):
        return "ABStat %s %s " % (len(self.st[self.sli]), ",".join(self.st.dtype.names) )
    brief = property(_get_brief)

    def __getitem__(self, sli):
        self.sli = sli
        return self

    def _get_ary(self, qwns=None):
        if qwns is None:
            qwns = list(self.QWNS)
        pass
        return np.vstack([self.st[q] for q in qwns]).T
    ary = property(_get_ary) 

    def chi2sel(self, cut=30):
        """
        :return indices: propagation record point indices with chi2 sum greater than cut
        """
        return np.where( np.sum(self.ar, axis=1) > cut )[0]

    def qctxsel(self, cut=30):
        """
        :return qctx list: were chi2 sum exceeds the cut 
        """ 
        return self.st.qctx[self.chi2sel(cut)]

    def reclabsel(self, cut=30):
        """
        :return reclab list: were chi2 sum exceeds the cut 
        """ 
        return self.st.reclab[self.chi2sel(cut)]


    @classmethod
    def dump(cls, st=None): 
        if st is None:
            st = cls.load()
        print st


if __name__ == '__main__':
    ok = opticks_main()
    st = ABStat.load()
    print st 
    ar = st.ar

          
