#!/usr/bin/env python
"""


CFH random access for debugging::

    delta:~ blyth$ ipython -i $(which cfh.py) -- /tmp/blyth/opticks/CFH/concentric/1/TO_BT_BT_BT_BT_SA/0/X
    Python 2.7.11 (default, Dec  5 2015, 23:51:51) 
    Type "copyright", "credits" or "license" for more information.

    IPython 1.2.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.
    /Users/blyth/opticks/ana/cfh.py /tmp/blyth/opticks/CFH/concentric/1/TO_BT_BT_BT_BT_SA/0/X
    ['/tmp/blyth/opticks/CFH/concentric/1/TO_BT_BT_BT_BT_SA/0/X']
    [2016-11-12 18:42:19,579] p4851 {/Users/blyth/opticks/ana/cfh.py:199} INFO - CFH.load from /tmp/blyth/opticks/CFH/concentric/1/TO_BT_BT_BT_BT_SA/0/X 

    In [1]: h
    Out[1]: X[0]

    In [2]: h.bins
    Out[2]: array([-76.3726, -61.0981, -45.8235, -30.549 , -15.2745,   0.    ,  15.2745,  30.549 ,  45.8235,  61.0981,  76.3726])


"""
import os, sys, json, logging, numpy as np
from opticks.ana.base import json_
from opticks.ana.ctx import Ctx 
from opticks.ana.nbase import chi2
from opticks.ana.abstat import ABStat
log = logging.getLogger(__name__)


class CFH(Ctx):
    """
    Persistable comparison histograms and chi2
    The members are numpy arrays and a single ctx dict
    allowing simple load/save.


    * :google:`chi2 comparison of normalized histograms`
    * http://www.hep.caltech.edu/~fcp/statistics/hypothesisTest/PoissonConsistency/PoissonConsistency.pdf


    """
    NAMES = "lhabc".split()


    @classmethod
    def load_(cls, ctx):
         if type(ctx) is list:
             return map(lambda _:cls.load_(_), ctx)
         elif type(ctx) is Ctx:
             h = CFH(ctx)  
             h.load()
             return h 
         else:
             log.warning("CFH.load_ unexpected ctx %s " % repr(ctx))
         return None

    def __init__(self, *args, **kwa):

        Ctx.__init__(self, *args, **kwa)

        #if type(ctx) is str:
        #    ctxs = Ctx.dir2ctx_(ctx)
        #    assert len(ctxs) == 1, "expect only a single ctx"
        #    ctx = ctxs[0]
        #pass

        # transients, not persisted
        self._log = False

    def __call__(self, bn, av, bv, lab, c2cut=30, c2shape=False):
        """
        :param bn: bin edges array
        :param av: a values array
        :param bv: b values array
        :param lab:
        :param c2cut: a+b stat requirement to compute chi2

        Called from AB.rhist

        """

        na = len(av)
        nb = len(bv)
        nv = 0.5*float(na + nb)

        #log.info("CFH.__call__ na %d nb %d nv %7.2f " % (na,nb,nv))

        ahis,_ = np.histogram(av, bins=bn)
        bhis,_ = np.histogram(bv, bins=bn)

        ah = ahis.astype(np.float32)
        bh = bhis.astype(np.float32)

        if c2shape:
            # shape comparison, normalize bin counts to average 
            #log.info("c2shape comparison")
            uah = ah*nv/float(na)
            ubh = bh*nv/float(nb)
        else:
            uah = ah 
            ubh = bh 
        pass

        c2, c2n, c2c = chi2(uah, ubh, cut=c2cut)

        assert len(ahis) == len(bhis) == len(c2)
        nval = len(ahis)
        assert len(bn) - 1 == nval

        lhabc = np.zeros((nval,5), dtype=np.float32)

        lhabc[:,0] = bn[0:-1] 
        lhabc[:,1] = bn[1:] 
        lhabc[:,2] = uah
        lhabc[:,3] = ubh
        lhabc[:,4] = c2

        self.lhabc = lhabc

        meta = {}
        meta['nedge'] = "%d" % len(bn)  
        meta['nval'] = "%d" % nval  

        meta['c2cut'] = c2cut  
        meta['c2n'] = c2n  
        meta['c2c'] = c2c 
        meta['la'] = lab[0] 
        meta['lb'] = lab[1] 

        meta['c2_ymax'] = "10"
        meta['logyfac'] = "3."
        meta['linyfac'] = "1.3"

        self.update(meta)


    @classmethod
    def c2per_(cls, hh):
        """
        :param hh: list of CFH instances
        :return c2per:  combined chi2 float 

        ::

            In [6]: hh = ab.hh

            In [7]: c2sums = map(lambda h:h.chi2.sum(), hh )
            Out[7]: [11.125423, 0.0, 0.0, 5.8574519, 180.07062, 182.38904, 208.7128, 11.125423]

            In [8]: c2nums = map(lambda h:h.c2n, hh )
            Out[8]: [19.0, 1.0, 1.0, 4.0, 222.0, 159.0, 218.0, 19.0]

            In [12]: sum(c2sums)
            Out[12]: 599.28075361251831

            In [13]: sum(c2nums)
            Out[13]: 643.0

            In [14]: sum(c2sums)/sum(c2nums)
            Out[14]: 0.93200739286550283

        """
        if type(hh) is CFH:
            hh = [hh]
        pass
        assert type(hh) is list 

        c2sums = map(lambda h:h.chi2.sum(), hh )
        c2nums = map(lambda h:h.c2n, hh )

        s_c2sums = sum(c2sums)
        s_c2nums = sum(c2nums)

        if s_c2nums > 0:
            c2per = s_c2sums/s_c2nums
        else:
            c2per = 0.
        pass
        return c2per


    ledg = property(lambda self:self.lhabc[:,0])
    hedg = property(lambda self:self.lhabc[:,1])
    ahis = property(lambda self:self.lhabc[:,2])
    bhis = property(lambda self:self.lhabc[:,3])
    chi2 = property(lambda self:self.lhabc[:,4])

    def _get_bins(self):
        """
        Recompose bins from lo and hi edges
        """
        lo = self.ledg
        hi = self.hedg

        bins = np.zeros(len(lo)+1, dtype=np.float32)
        bins[0:-1] = lo
        bins[-1] = hi[-1]

        return bins
    bins = property(_get_bins)


    def _get_ndf(self):
        ndf = max(self.c2n - 1, 1)
        return ndf 
    ndf = property(_get_ndf)

    def _get_c2p(self):
        ndf = self.ndf 
        c2p = self.chi2.sum()/ndf
        return c2p 
    c2p = property(_get_c2p)
    
    def _get_c2label(self):   
        return "chi2/ndf %4.2f [%d]" % (self.c2p, self.ndf)
    c2label = property(_get_c2label)


    def _set_log(self, log_=True):
        self._log = log_
    def _get_log(self):
        return self._log
    log = property(_get_log, _set_log) 

    def _get_ylim(self):
        ymin = 1 if self.log else 0 
        yfac = self.logyfac if self.log else self.linyfac
        ymax = max(self.ahis.max(), self.bhis.max()) 
        return [ymin,ymax*yfac]
    ylim = property(_get_ylim)


    def _get_ctxstr(self, name, fallback="?"):
        return str(self.get(name,fallback))

    def _get_ctxfloat(self, name, fallback="0"):
        return float(self.get(name,fallback))

    def _get_ctxint(self, name, fallback="0"):
        return int(self.get(name,fallback))


    #seq0 = property(lambda self:self.ctx.get("seq0", None))
    la = property(lambda self:self._get_ctxstr("la"))
    lb = property(lambda self:self._get_ctxstr("lb"))
    qwn = property(lambda self:self._get_ctxstr("qwn"))

    c2_ymax = property(lambda self:self._get_ctxfloat("c2_ymax"))
    logyfac = property(lambda self:self._get_ctxfloat("logyfac"))
    linyfac = property(lambda self:self._get_ctxfloat("linyfac"))
    c2n = property(lambda self:self._get_ctxfloat("c2n"))
    c2c = property(lambda self:self._get_ctxfloat("c2c"))
    c2cut = property(lambda self:self._get_ctxfloat("c2cut"))

    nedge = property(lambda self:self._get_ctxint("nedge"))
    nval = property(lambda self:self._get_ctxint("nval"))
    irec = property(lambda self:self._get_ctxint("irec"))

    def __repr__(self):
        return "%s[%s]" % (self.qwn,self.irec)

    def ctxpath(self):
        return self.path("ctx.json") 

    def paths(self):
        return [self.ctxpath()] + map(lambda name:self.path(name+".npy"), self.NAMES)

    def exists(self):
        if self.seq0 is None:
            log.warning("CFH.exists can only be used with single line selections")
            return False 

        paths = self.paths()
        a = np.zeros(len(paths), dtype=np.bool)
        for i,path in enumerate(paths):
            a[i] = os.path.exists(path)
        return np.all(a[i] == True)

    def save(self):
        if self.seq0 is None:
            log.warning("CFH.save can only be used with single line selections")
            return  

        dir_ = self.dir
        log.debug("CFH.save to %s " % dir_)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        json.dump(dict(self), file(self.ctxpath(),"w") )
        for name in self.NAMES:
            np.save(self.path(name+".npy"), getattr(self, name))
        pass
 

    def load(self):
        if self.seq0 is None:
            log.warning("CFH.load can only be used with single line selections")
            return  

        dir_ = self.dir
        log.debug("CFH.load from %s " % dir_)

        exists = os.path.exists(dir_)
        if not exists:
            log.fatal("CFH.load non existing dir %s  " % (dir_) )

        assert exists, dir_

        js = json_(self.ctxpath())
        k = map(str, js.keys())
        v = map(str, js.values())

        self.update(dict(zip(k,v)))

        for name in self.NAMES:
            setattr(self, name, np.load(self.path(name+".npy")))
        pass



def test_load():
    ctx = {'det':"concentric", 'tag':"1", 'qwn':"X", 'irec':"5", 'seq0':"TO_BT_BT_BT_BT_DR_SA" }
    h = CFH_.load(ctx)



if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    ok = opticks_main()
    print ok

    import matplotlib.pyplot as plt
    plt.rcParams["figure.max_open_warning"] = 200    # default is 20
    plt.ion()


    from opticks.ana.ab import AB
    from opticks.ana.cfplot import one_cfplot, qwns_plot 
    print ok.nargs

    ## only reload evts for rehisting
    if ok.rehist:
        ab = AB(ok)
    else:
        ab = None
    pass


    st = ABStat.load(ok)

    if ok.chi2sel:
        reclabs = st.reclabsel()
    elif len(ok.nargs) > 0:
        reclabs = [ok.nargs[0]]
    else:
        reclabs = ["[TO] AB",]
    pass

    n_reclabs = len(reclabs)
    log.info(" n_reclabs : %d " % (n_reclabs))

    n_limit = 50 
    if n_reclabs > n_limit:
        log.warning("too many reclabs truncating to %d" % n_limit )  
        reclabs = reclabs[:n_limit]
    pass

    for reclab in reclabs:

        ctx = Ctx.reclab2ctx_(reclab, det=ok.det, tag=ok.tag)

        st[st.st.reclab==reclab]   # sets a single line slice
        suptitle = st.suptitle

        ctxs = ctx.qsub()
        assert len(ctxs) == 8 , ctxs
        log.info(" %s " % suptitle)

        if ok.rehist:
            hh = ab.rhist_(ctxs, rehist=True)
        else:
            hh = CFH.load_(ctxs)
        pass

        if len(hh) == 1:
            one_cfplot(ok, hh[0])
        else:
            qwns_plot(ok, hh, suptitle  )
        pass
    pass



