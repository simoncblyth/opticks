#!/usr/bin/env python
"""
qcf.py
======


"""
import os, logging, numpy as np
from opticks.ana.nbase import chi2, chi2_pvalue
log = logging.getLogger(__name__)

class QU(object):
    def __init__(self, q, symbol="q"):
        """
        QU : Unique value table 

        u : unique values within q
        x : first index within q of the unique value
        n : count of occurrence of the unique values in q 
        """

        expr = "np.c_[n,x,u][o][lim]"
        label = "uniques in descending count order with first index x"

        u, x, n = np.unique(q, return_index=True, return_counts=True)
        o = np.argsort(n)[::-1]  
        # HMM:notice that the ordering is not applied here 

        self.symbol = symbol
        self.expr = expr 
        self.label = label

        self.q = q 
        self.uxno = u,x,n,o
        self.u = u 
        self.x = x 
        self.n = n 
        self.o = o 
        self.lim = slice(0,10)

    def _get_tab(self):
        u,x,n,o = self.uxno
        expr = self.expr
        lim = self.lim 
        return eval(expr)

    tab = property(_get_tab)

    def __repr__(self):
        lines = []
        lines.append("%s : %s : %s" % (self.symbol, self.expr, self.label))
        lines.append(str(self.tab))
        return "\n".join(lines)


class QCF(object):
    """
    Used for example from sevt.py:SAB
    """
    def __init__(self, _aq, _bq, symbol="qcf"):
        """
        :param _aq: photon history strings for event A
        :param _bq: ditto for B 
        """
        _same_shape = _aq.shape == _bq.shape
        if not _same_shape:
            mn = min(_aq.shape[0], _bq.shape[0])
            msg = "a, b not same shape _aq %s _bq %s WILL LIMIT TO mn %d " % ( str(_aq.shape), str(_bq.shape), mn ) 
            lim = slice(0, mn)
        else:
            lim = slice(None)
            msg = ""
        pass
        aq = _aq[lim]
        bq = _bq[lim]
        same_shape = aq.shape == bq.shape
        assert same_shape

        asym = "%s.aqu" % symbol

        log.info("QCF.__init__ %s " % asym)  
        aqu = QU(aq, symbol=asym )
        bsym = "%s.bqu" % symbol
        log.info("QCF.__init__ %s " % bsym)  
        bqu = QU(bq, symbol=bsym ) 

        qu = np.unique(np.concatenate([aqu.u,bqu.u]))       ## unique histories of both A and B in uncontrolled order
        ab = np.zeros( (len(qu),3,2), dtype=np.int64 )

        log.info("QCF.__init__ [ qu loop")
        for i, q in enumerate(qu):
            ai_ = np.where(aqu.u == q )[0]           # find indices in the a and b unique lists 
            bi_ = np.where(bqu.u == q )[0]
            ai = ai_[0] if len(ai_) == 1 else -1
            bi = bi_[0] if len(bi_) == 1 else -1

            # NB the ai and bi are internal indices into the separate A and B lists
            # so they are necessary but not ordinarily surfaced 
            # as not very human digestible 
            #
            # effectively ai and bi are pointers into the two unique lists
           
            if i % 1000 == 0:
                log.info("QCF.__init__ . qu loop %d " % i )
            pass

            ab[i,0,0] = ai
            ab[i,1,0] = aqu.x[ai] if ai > -1 else -1  ## index of first occurrence
            ab[i,2,0] = aqu.n[ai] if ai > -1 else 0   ## count in each 

            ab[i,0,1] = bi
            ab[i,1,1] = bqu.x[bi] if bi > -1 else -1
            ab[i,2,1] = bqu.n[bi] if bi > -1 else 0
        pass
        log.info("QCF.__init__ ] qu loop")

        abx = np.max(ab[:,2,:], axis=1 )   # max of aqn, bqn counts 
        abxo = np.argsort(abx)[::-1]       # descending count order indices
        abo = ab[abxo]                     # ab ordered  
        quo = qu[abxo]                     # qu ordered 
        iq = np.arange(len(qu))

        # more than 10 counts in one, but zero in the other : history dropouts are smoking guns for bugs 
        bzero = np.where( np.logical_and( abo[:,2,0] > 10, abo[:,2,1] == 0 ) )[0]
        azero = np.where( np.logical_and( abo[:,2,1] > 10, abo[:,2,0] == 0 ) )[0]

        c2cut = int(os.environ.get("C2CUT","30"))
        c2,c2n,c2c = chi2( abo[:,2,0], abo[:,2,1], cut=c2cut )
        c2sum = c2.sum()
        c2per = c2sum/c2n

        c2pv = chi2_pvalue( c2sum, int(c2n) )
        c2pvm = "> 0.05 : null-hyp " if c2pv > 0.05 else "< 0.05 : NOT:null-hyp "  
        c2pvd = "pv[%4.3f,%s] " % (c2pv, c2pvm)
        # null-hyp consistent means there is no significant difference between 
        # the frequency counts in the A and B samples at a certain confidence
        # level (normally 5%) 
        
        c2desc = "c2sum/c2n:c2per(C2CUT)  %5.2f/%d:%5.3f (%2d) %s" % ( c2sum, int(c2n), c2per, c2cut, c2pvd )
        c2label = "c2sum : %10.4f c2n : %10.4f c2per: %10.4f  C2CUT: %4d " % ( c2sum, c2n, c2per, c2cut )



        self._aq = _aq
        self._bq = _bq
        self.lim = lim 
        self.msg = msg 
        self.aq = aq
        self.bq = bq
        self.symbol = symbol
        self.aqu = aqu
        self.bqu = bqu
        self.qu = qu
        self.ab = ab
        self.abx = abx
        self.abxo = abxo
        self.abo = abo
        self.quo = quo
        self.iq = iq
        self.bzero = bzero
        self.azero = azero
        self.c2cut = c2cut
        self.c2 = c2
        self.c2n = c2n
        self.c2c = c2c
        self.c2sum = c2sum
        self.c2per = c2per
        self.c2desc = c2desc
        self.c2label = c2label
        self.sli = slice(None)

    def __getitem__(self, sli):
        """

        Change slice with::

           ab.qcf[:100]  

        """
        self.sli = sli
        print("sli: %s " % str(sli))
        return self
 
    def __repr__(self):
        lines = []
        lines.append("QCF %s : %s " % (self.symbol, self.msg))
        lines.append("a.q %d b.q %d lim %s " % (self._aq.shape[0], self._bq.shape[0], str(self.lim)) )
        lines.append(self.c2label)
        lines.append(self.c2desc)

        siq = list(map(lambda _:"%2d" % _ , self.iq ))  # row index 
        sc2 = list(map(lambda _:"%7.4f" % _, self.c2 ))

        sabo2 = list(map(lambda _:"%6d %6d" % tuple(_), self.abo[:,2,:]))
        sabo1 = list(map(lambda _:"%6d %6d" % tuple(_), self.abo[:,1,:]))

        pstr_ = lambda _:_.strip().decode("utf-8")
        _quo = list(map(pstr_, self.quo))
        mxl = max(list(map(len, _quo)))
        fmt = "%-" + str(mxl) + "s"
        _quo = list(map(lambda _:fmt % _, _quo ))
        _quo = np.array( _quo )

        sli = self.sli

        start = sli.start if not sli.start is None else 0 
        stop = sli.stop if not sli.stop is None else 25
        step = sli.step if not sli.step is None else None  # not used

        #abexpr = "np.c_[quo,abo[:,2,:],abo[:,1,:]]"
        abexpr = "np.c_[siq,_quo,siq,sabo2,sc2,sabo1]"
        subs = "[%d:%d]" % ( start, stop ) 
        subs += " [bzero] [azero]"
        subs = subs.split()
        descs = ["A-B history frequency chi2 comparison", "in A but not B", "in B but not A" ]

        bzero = self.bzero
        azero = self.azero

        for i in range(len(subs)):
            expr = "%s%s" % (abexpr, subs[i])
            lines.append("\n%s  ## %s " % (expr, descs[i]) )
            lines.append(str(eval(expr)))
        pass
        return "\n".join(lines)

class QCFZero(object):
    """
    Presenting dropout histories in comparison : smoking gun for bugs 
    """
    def __init__(self, qcf, symbol="qcf0"):
        self.qcf = qcf
        self.bzero_viz = "u4t ; N=0 APID=%d AOPT=idx ./U4SimtraceTest.sh ana"
        self.azero_viz = "u4t ; N=1 BPID=%d BOPT=idx ./U4SimtraceTest.sh ana"

    def __repr__(self):
        qcf = self.qcf
        quo = qcf.quo
        aq = qcf.aq
        bq = qcf.bq  
        bzero = qcf.bzero
        azero = qcf.azero

        pstr_ = lambda _:_.strip().decode("utf-8")

        lines = []
        lim = slice(0,2)
        lines.append("\nbzero : %s : A HIST NOT IN B" % (str(bzero)))
        for _ in bzero:
            idxs = np.where( quo[_] == aq[:,0] )[0]
            lines.append("bzero quo[_]:%s len(idxs):%d idxs[lim]:%s " % ( pstr_(quo[_]), len(idxs), str(idxs[lim])) )
            for idx in idxs[lim]:
                lines.append(self.bzero_viz % idx)
            pass
            if len(idxs) > 0: lines.append("")
        pass

        lines.append("\nazero : %s : B HIST NOT IN A" % (str(azero)))
        for _ in azero:
            idxs = np.where( quo[_] == bq[:,0] )[0]
            lines.append("azero quo[_]:%s len(idxs):%d idxs[lim]:%s " % ( pstr_(quo[_]), len(idxs), str(idxs[lim])) )
            for idx in idxs[lim]:
                lines.append(self.azero_viz % idx)
            pass
            if len(idxs) > 0: lines.append("")
        pass
        return "\n".join(lines)


def test_QU():
    aq = np.array( ["red", "green", "blue", "blue" ], dtype="|S10" )
    aqu = QU(aq, symbol="aqu")
    print(aqu.tab)
    print(aqu)

def test_QCF():
    aq = np.array( ["red", "green", "blue", "blue" ], dtype="|S10" )
    bq = np.array( ["red", "red", "green", "blue"  ], dtype="|S10" )

    qcf = QCF(aq, bq, symbol="qcf")
    print(qcf.ab)
    print(qcf)





if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    aq = np.array( ["red", "green", "blue", "cyan" ], dtype="|S10" )
    bq = np.array( ["red", "red", "green", "blue"  ], dtype="|S10" )

    qcf = QCF(aq, bq, symbol="qcf")
    print(qcf.ab)
    print(qcf)

    qcf0 = QCFZero(qcf, symbol="qcf0")
    print(qcf0)






